import math
import os
import torch
import weakref
from functools import lru_cache
from torch.utils._triton import has_triton
from ._triton_ops_meta import get_meta
from typing import Optional, Tuple
def _scatter_mm6(blocks: torch.Tensor, others: torch.Tensor, c_indices: torch.Tensor, r_offsets: torch.Tensor, p_offsets: torch.Tensor, q_offsets: torch.Tensor, meta: dict, accumulators: torch.Tensor, force_contiguous: bool=True):
    SPLIT_N = meta['SPLIT_N']
    P, Ms, Ks = blocks.shape
    B, K_, N = others.shape
    B_, M, N_ = accumulators.shape
    assert N_ == N
    Ns = N // SPLIT_N
    assert B_ == B

    def grid(META):
        return (r_offsets.shape[0] * B, triton.cdiv(Ms, META['TILE_M']) * triton.cdiv(Ns, META['TILE_N']))
    dot_out_dtype = {torch.float16: tl.float32, torch.bfloat16: tl.float32, torch.float32: tl.float64, torch.float64: tl.float64}[accumulators.dtype]
    if 'allow_tf32' not in meta:
        meta.update(allow_tf32=dot_out_dtype == tl.float32)
    assert c_indices.stride(0) == 1
    assert r_offsets.stride(0) == 1
    assert p_offsets.stride(0) == 1
    assert q_offsets.stride(0) == 1
    if force_contiguous:
        blocks = blocks.contiguous()
        others = others.contiguous()
        if not accumulators.is_contiguous():
            accumulators_ = accumulators.contiguous()
        else:
            accumulators_ = accumulators
    else:
        accumulators_ = accumulators
    _scatter_mm6_kernel[grid](B, Ms, Ks, N, blocks, blocks.stride(0), blocks.stride(1), blocks.stride(2), others, others.stride(0), others.stride(1), others.stride(2), accumulators_, accumulators_.stride(0), accumulators_.stride(1), accumulators_.stride(2), c_indices, r_offsets, p_offsets, q_offsets, dot_out_dtype=dot_out_dtype, **meta)
    if force_contiguous and (not accumulators.is_contiguous()):
        accumulators.copy_(accumulators_)