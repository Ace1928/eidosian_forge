import math
import os
import torch
import weakref
from functools import lru_cache
from torch.utils._triton import has_triton
from ._triton_ops_meta import get_meta
from typing import Optional, Tuple
def _run_sampled_addmm_kernel(alpha, beta, is_beta_zero, blocksize, k, tile_k, values, crow_indices, col_indices, mat1, mat2, max_grid):
    n_batches = values.size(0)
    n_block_rows = crow_indices.size(-1) - 1
    full_grid = (n_batches, n_block_rows)
    if max_grid is not None:
        grid_blocks = tuple(max_grid[:2][::-1]) + (None,) * (2 - len(max_grid[:2]))
    else:
        grid_blocks = None
    tensor_dims_map = {values: (0, None), crow_indices: (0, -1), col_indices: (0, None), mat1: (0, -4), mat2: (0, None)}
    if values.dtype in (torch.half, torch.bfloat16):
        acc_dtype = tl.float32
        allow_tf32 = True
    else:
        acc_dtype = tl.float64
        allow_tf32 = False

    def kernel(grid, *sliced_tensors):
        _sampled_addmm_kernel[grid](alpha, beta, is_beta_zero, *blocksize, k, tile_k, *ptr_stride_extractor(*sliced_tensors), acc_dtype=acc_dtype, allow_tf32=allow_tf32, num_stages=1, num_warps=4)
    launch_kernel(kernel, tensor_dims_map, full_grid, grid_blocks)