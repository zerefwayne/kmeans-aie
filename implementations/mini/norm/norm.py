# vector_reduce_add/vector_reduce_add_placed.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_


def my_reduce_add():
    N_ROWS = 32
    N_COLS = 16
    
    IN0_VOLUME = N_ROWS * N_COLS
    IN1_VOLUME = N_ROWS

    buffer_depth = 2
    dev = AIEDevice.npu1_1col

    @device(dev)
    def device_body():
        in_ty = np.ndarray[(IN0_VOLUME,), np.dtype[np.int32]]
        out_ty = np.ndarray[(IN1_VOLUME,), np.dtype[np.int32]]

        # AIE Core Function declarations
        compute_norm = external_func(
            "compute_norm", 
            inputs=[in_ty, out_ty, np.int32, np.int32]
        )

        # Tile declarations
        ShimTile = tile(0, 0)
        ComputeTile2 = tile(0, 2)

        # AIE-array data movement with object fifos
        of_in = object_fifo("in", ShimTile, ComputeTile2, buffer_depth, in_ty)
        of_out = object_fifo("out", ComputeTile2, ShimTile, buffer_depth, out_ty)

        # Set up compute tiles

        # Compute tile 2
        @core(ComputeTile2, "norm.cc.o")
        def core_body():
            for _ in range_(0xFFFFFFFF):
                elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                compute_norm(elem_in, elem_out, N_ROWS, N_COLS)
                of_in.release(ObjectFifoPort.Consume, 1)
                of_out.release(ObjectFifoPort.Produce, 1)

        # To/from AIE-array data movement
        @runtime_sequence(in_ty, out_ty)
        def sequence(A, C):
            in_task = shim_dma_single_bd_task(of_in, A, sizes=[1, 1, 1, IN0_VOLUME])
            out_task = shim_dma_single_bd_task(
                of_out, C, sizes=[1, 1, 1, IN1_VOLUME], issue_token=True
            )

            dma_start_task(in_task, out_task)
            dma_await_task(out_task)
            dma_free_task(in_task)


with mlir_mod_ctx() as ctx:
    my_reduce_add()
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
