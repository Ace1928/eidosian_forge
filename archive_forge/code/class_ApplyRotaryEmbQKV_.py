import math
from typing import Optional, Tuple, Union
import torch
from einops import rearrange, repeat
from flash_attn.ops.triton.rotary import apply_rotary
class ApplyRotaryEmbQKV_(torch.autograd.Function):

    @staticmethod
    def forward(ctx, qkv, cos, sin, cos_k=None, sin_k=None, interleaved=False, seqlen_offsets: Union[int, torch.Tensor]=0):
        batch, seqlen, three, nheads, headdim = qkv.shape
        assert three == 3
        if cos_k is None and sin_k is None and qkv.is_contiguous():
            qk = qkv[:, :, :2].reshape(batch, seqlen, -1, headdim)
            apply_rotary(qk, cos, sin, seqlen_offsets=seqlen_offsets, interleaved=interleaved, inplace=True)
        else:
            cos_k = cos if cos_k is None else cos_k
            sin_k = sin if sin_k is None else sin_k
            q, k = (qkv[:, :, 0], qkv[:, :, 1])
            apply_rotary(q, cos, sin, seqlen_offsets, interleaved=interleaved, inplace=True)
            apply_rotary(k, cos_k, sin_k, seqlen_offsets, interleaved=interleaved, inplace=True)
            ctx.save_for_backward(cos, sin, cos_k, sin_k)
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin, cos_k, sin_k)
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cos_k, sin_k, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        return qkv

    @staticmethod
    def backward(ctx, dqkv):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cos_k, sin_k, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cos_k, sin_k = ctx.saved_tensors
        if cos_k is None and sin_k is None and dqkv.is_contiguous():
            dqk = rearrange(dqkv[:, :, :2], 'b s t h d -> b s (t h) d')
            apply_rotary(dqk, cos, sin, seqlen_offsets=seqlen_offsets, interleaved=ctx.interleaved, inplace=True, conjugate=True)
        else:
            cos_k = cos if cos_k is None else cos_k
            sin_k = sin if sin_k is None else sin_k
            dq, dk = (dqkv[:, :, 0], dqkv[:, :, 1])
            apply_rotary(dq, cos, sin, seqlen_offsets, interleaved=ctx.interleaved, inplace=True, conjugate=True)
            apply_rotary(dk, cos_k, sin_k, seqlen_offsets, interleaved=ctx.interleaved, inplace=True, conjugate=True)
        return (dqkv, None, None, None, None, None, None)