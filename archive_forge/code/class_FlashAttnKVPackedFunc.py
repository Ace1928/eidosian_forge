import math
import torch
import triton
import triton.language as tl
class FlashAttnKVPackedFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, kv, bias=None, causal=False, softmax_scale=None):
        """
        q: (batch, seqlen_q, nheads, headdim)
        kv: (batch, seqlen_k, 2, nheads, headdim)
        bias: optional, shape broadcastible to (batch, nheads, seqlen_q, seqlen_k).
            For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen_k).
            ALiBi mask for non-causal would have shape (1, nheads, seqlen_q, seqlen_k)
        """
        q, kv = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, kv]]
        o, lse, ctx.softmax_scale = _flash_attn_forward(q, kv[:, :, 0], kv[:, :, 1], bias=bias, causal=causal, softmax_scale=softmax_scale)
        ctx.save_for_backward(q, kv, o, lse, bias)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, kv, o, lse, bias = ctx.saved_tensors
        if len(ctx.needs_input_grad) >= 3:
            assert not ctx.needs_input_grad[2], 'FlashAttention does not support bias gradient yet'
        with torch.inference_mode():
            dq = torch.empty_like(q)
            dkv = torch.empty_like(kv)
            _flash_attn_backward(do, q, kv[:, :, 0], kv[:, :, 1], o, lse, dq, dkv[:, :, 0], dkv[:, :, 1], bias=bias, causal=ctx.causal, softmax_scale=ctx.softmax_scale)
        return (dq, dkv, None, None, None)