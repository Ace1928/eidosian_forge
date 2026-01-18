from typing import Any, Optional, Sequence, Tuple, Type, Union
import torch
from . import (
from .attn_bias import (
from .common import (
from .dispatch import _dispatch_bw, _dispatch_fw, _ensure_op_supports_or_raise
class _fMHA(torch.autograd.Function):

    @staticmethod
    def forward(ctx, op: AttentionOp, *args: Any) -> Any:
        inp = Inputs(*args)
        op_fw = op[0] if op is not None else None
        op_bw = op[1] if op is not None else None
        out, op_ctx = _memory_efficient_attention_forward_requires_grad(inp=inp, op=op_fw)
        if isinstance(inp.attn_bias, torch.Tensor):
            attn_bias_tensor = inp.attn_bias
            attn_bias_ctx = None
        else:
            attn_bias_tensor = None
            attn_bias_ctx = inp.attn_bias
        ctx.save_for_backward(inp.query, inp.key, inp.value, op_ctx.out, op_ctx.lse)
        ctx.rng_state = op_ctx.rng_state
        ctx.attn_bias_tensor = attn_bias_tensor
        if op_ctx.op_bw is not None:
            if op_bw is not None and op_bw is not op_ctx.op_bw:
                raise ValueError(f'Specified op_bw={op_bw.NAME}, but forward op can only run with op_bw={op_ctx.op_bw.NAME}. Please set op_bw=None.')
            op_bw = op_ctx.op_bw
        ctx.op_fw = op_fw
        ctx.op_bw = op_bw
        ctx.p = inp.p
        ctx.scale = inp.scale
        ctx.attn_bias_ctx = attn_bias_ctx
        ctx.n_args = len(args)
        return out

    @staticmethod
    def deserialize_bias(attn_bias_ctx, attn_bias_tensor: Optional[torch.Tensor]) -> Any:
        if attn_bias_tensor is None:
            return attn_bias_ctx
        return attn_bias_tensor

    @classmethod
    @torch.autograd.function.once_differentiable
    def backward(cls, ctx, grad):
        query, key, value, out, lse = ctx.saved_tensors
        attn_bias_tensor = ctx.attn_bias_tensor
        rng_state = ctx.rng_state
        inp = Inputs(query=query, key=key, value=value, attn_bias=cls.deserialize_bias(ctx.attn_bias_ctx, attn_bias_tensor), p=ctx.p, scale=ctx.scale)
        op_ctx = Context(lse=lse, out=out, rng_state=rng_state)
        grads = _memory_efficient_attention_backward(ctx=op_ctx, inp=inp, grad=grad, op=ctx.op_bw)
        return (None, grads.dq, grads.dk, grads.dv, grads.db) + (None,) * (ctx.n_args - 2)