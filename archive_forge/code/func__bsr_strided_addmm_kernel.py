import math
import os
import torch
import weakref
from functools import lru_cache
from torch.utils._triton import has_triton
from ._triton_ops_meta import get_meta
from typing import Optional, Tuple
@triton.jit
def _bsr_strided_addmm_kernel(values_ptr, values_batch_stride, values_nnz_stride, values_row_block_stride, values_col_block_stride, crow_indices_ptr, crow_indices_batch_stride, crow_indices_stride, col_indices_ptr, col_indices_batch_stride, col_indices_stride, input_ptr, input_batch_stride, input_tiled_row_stride, input_tiled_col_stride, input_row_block_stride, input_col_block_stride, dense_ptr, dense_batch_stride, dense_tiled_row_stride, dense_tiled_col_stride, dense_row_block_stride, dense_col_block_stride, output_ptr, output_batch_stride, output_tiled_row_stride, output_tiled_col_stride, output_row_block_stride, output_col_block_stride, beta, alpha, beta_is_one: tl.constexpr, beta_is_nonzero: tl.constexpr, alpha_is_one: tl.constexpr, BLOCKSIZE_ROW: tl.constexpr, BLOCKSIZE_COL: tl.constexpr, BLOCKSIZE_INNER: tl.constexpr, acc_dtype: tl.constexpr, allow_tf32: tl.constexpr, GROUP_SIZE_ROW: tl.constexpr, SPLIT_N: tl.constexpr):
    batch_pid = tl.program_id(axis=2)
    row_block_pid = tl.program_id(axis=0)
    col_block_pid = tl.program_id(axis=1)
    n_block_rows = tl.num_programs(axis=0)
    n_block_cols = tl.num_programs(axis=1)
    row_block_pid, col_block_pid = tl.swizzle2d(row_block_pid, col_block_pid, n_block_rows, n_block_cols, GROUP_SIZE_ROW)
    crow_indices_offset_ptr = crow_indices_ptr + crow_indices_batch_stride * batch_pid + crow_indices_stride * row_block_pid
    nnz_offset = tl.load(crow_indices_offset_ptr)
    nnz_offset_next = tl.load(crow_indices_offset_ptr + crow_indices_stride)
    row_nnz = nnz_offset_next - nnz_offset
    row_block_arange = tl.arange(0, BLOCKSIZE_ROW)
    inner_block_arange = tl.arange(0, BLOCKSIZE_INNER)
    col_block_arange = tl.arange(0, BLOCKSIZE_COL)
    input_ptrs = input_ptr + input_batch_stride * batch_pid + input_tiled_row_stride * row_block_pid + input_tiled_col_stride * col_block_pid + input_row_block_stride * row_block_arange[:, None] + input_col_block_stride * col_block_arange[None, :]
    values_block_ptrs = values_ptr + values_batch_stride * batch_pid + values_nnz_stride * nnz_offset + values_row_block_stride * row_block_arange[:, None] + values_col_block_stride * inner_block_arange[None, :]
    dense_block_ptrs = dense_ptr + dense_batch_stride * batch_pid + dense_tiled_col_stride * col_block_pid + dense_row_block_stride * inner_block_arange[:, None] + dense_col_block_stride * col_block_arange[None, :]
    output_ptrs = output_ptr + output_batch_stride * batch_pid + output_tiled_row_stride * row_block_pid + output_tiled_col_stride * col_block_pid + output_row_block_stride * row_block_arange[:, None] + output_col_block_stride * col_block_arange[None, :]
    col_index_nnz_ptr = col_indices_ptr + col_indices_batch_stride * batch_pid + col_indices_stride * nnz_offset
    if beta_is_nonzero:
        output_acc_block = tl.load(input_ptrs).to(acc_dtype)
        if not (beta_is_one and alpha_is_one):
            beta_alpha = beta / alpha
            output_acc_block *= beta_alpha
    else:
        output_acc_block = tl.zeros((BLOCKSIZE_ROW, BLOCKSIZE_COL), dtype=acc_dtype)
    for _ in range(row_nnz):
        values_block = tl.load(values_block_ptrs)
        dense_row_idx = tl.load(col_index_nnz_ptr)
        dense_block = tl.load(dense_block_ptrs + dense_tiled_row_stride * dense_row_idx)
        output_acc_block += tl.dot(values_block, dense_block, allow_tf32=allow_tf32, out_dtype=acc_dtype)
        values_block_ptrs += values_nnz_stride
        col_index_nnz_ptr += col_indices_stride
    if not alpha_is_one:
        output_acc_block *= alpha
    tl.store(output_ptrs, output_acc_block.to(output_ptr.dtype.element_ty))