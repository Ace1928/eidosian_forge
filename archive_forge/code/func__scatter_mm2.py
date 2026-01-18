import math
import os
import torch
import weakref
from functools import lru_cache
from torch.utils._triton import has_triton
from ._triton_ops_meta import get_meta
from typing import Optional, Tuple
def _scatter_mm2(blocks: torch.Tensor, others: torch.Tensor, pq_offsets: torch.Tensor, pq_indices: torch.Tensor, accumulators: torch.Tensor):
    P, M, K = blocks.shape
    Q, _, N = others.shape
    R, _, _ = accumulators.shape
    meta = dict(TILE_M=max(16, M // 4), TILE_N=max(16, N // 4), num_stages=1, num_warps=2)

    def grid(META):
        return (pq_offsets.shape[0] - 1, triton.cdiv(M, META['TILE_M']) * triton.cdiv(N, META['TILE_N']), 1)
    dot_out_dtype = {torch.float16: tl.float32, torch.bfloat16: tl.float32, torch.float32: tl.float64, torch.float64: tl.float64}[accumulators.dtype]
    if 'allow_tf32' not in meta:
        meta.update(allow_tf32=dot_out_dtype == tl.float32)
    _scatter_mm2_kernel[grid](M, K, N, blocks, blocks.stride(0), blocks.stride(1), blocks.stride(2), others, others.stride(0), others.stride(1), others.stride(2), accumulators, accumulators.stride(0), accumulators.stride(1), accumulators.stride(2), pq_offsets, pq_offsets.stride(0), pq_indices, pq_indices.stride(0), pq_indices.stride(1), dot_out_dtype=dot_out_dtype, **meta)