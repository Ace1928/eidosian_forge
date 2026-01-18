import functools
import sys
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Type
import torch
from ..common import _has_triton21, register_operator
from .attn_bias import (
from .common import AttentionFwOpBase, Context, Inputs, check_lastdim_alignment_stride1
@triton.jit
def _splitK_reduce(Out_splitK, LSE_splitK, Out, LSE, split_k: tl.constexpr, stride_osk_z, stride_osk_g, stride_osk_h, stride_osk_s, stride_osk_m, stride_osk_k, stride_lsek_z, stride_lsek_g, stride_lsek_h, stride_lsek_s, stride_lsek_m, stride_oz, stride_og, stride_oh, stride_om, stride_ok, stride_lse_z, stride_lse_g, stride_lse_h, stride_lse_m, BLOCK_SIZE: tl.constexpr, H: tl.constexpr, G: tl.constexpr, WRITE_LSE: tl.constexpr):
    off_m = tl.program_id(0).to(tl.int64)
    off_z = tl.program_id(1).to(tl.int64)
    off_gh = tl.program_id(2).to(tl.int64)
    off_g = off_gh % G
    off_h = off_gh // G
    Out_splitK_ptr = Out_splitK + stride_osk_z * off_z + stride_osk_g * off_g + stride_osk_h * off_h + stride_osk_m * off_m + tl.arange(0, BLOCK_SIZE)
    LSE_splitK_ptr0 = LSE_splitK + stride_lsek_z * off_z + stride_lsek_g * off_g + stride_lsek_h * off_h + stride_lsek_m * off_m
    LSE_splitK_ptr = LSE_splitK_ptr0
    lse_max = tl.load(LSE_splitK_ptr)
    for split_k_idx in tl.static_range(1, split_k):
        LSE_splitK_ptr = LSE_splitK_ptr + stride_lsek_s
        lse_splitk = tl.load(LSE_splitK_ptr)
        lse_max = tl.maximum(lse_max, lse_splitk)
    sumexp_normalized = 0.0
    numerator_normalized = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    LSE_splitK_ptr = LSE_splitK_ptr0
    for split_k_idx in tl.static_range(0, split_k):
        out_splitk = tl.load(Out_splitK_ptr)
        lse_splitk = tl.load(LSE_splitK_ptr)
        sumexp_normalized_splitk = tl.math.exp2((lse_splitk - lse_max).to(tl.float32) * 1.44269504)
        sumexp_normalized += sumexp_normalized_splitk
        numerator_normalized += out_splitk * sumexp_normalized_splitk
        LSE_splitK_ptr = LSE_splitK_ptr + stride_lsek_s
        Out_splitK_ptr = Out_splitK_ptr + stride_osk_s
    acc = numerator_normalized / sumexp_normalized
    Out_ptr = Out + stride_oz * off_z + stride_oh * off_h + stride_og * off_g + stride_om * off_m + tl.arange(0, BLOCK_SIZE)
    if acc.dtype is tl.float64 and Out.dtype.element_ty is not tl.float64:
        acc = acc.to(tl.float32)
    tl.store(Out_ptr, acc)
    if WRITE_LSE:
        l_ptrs = LSE + off_z * stride_lse_z + off_g * stride_lse_g + off_h * stride_lse_h + off_m * stride_lse_m
        tl.store(l_ptrs, lse_max + tl.math.log2(sumexp_normalized) / 1.44269504)