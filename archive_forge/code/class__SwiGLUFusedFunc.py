from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from .common import BaseOperator, get_xformers_operator, register_operator
from .unbind import stack_or_none, unbind
class _SwiGLUFusedFunc(torch.autograd.Function):
    NAME = 'fused.py'

    @classmethod
    @torch.cuda.amp.custom_fwd
    def forward(cls, ctx, x, w1, b1, w2, b2, w3, b3):
        x1, x2, x4 = DualGemmSiluOp.OPERATOR(x, w1, b1, w2, b2)
        x5 = F.linear(x4, w3, b3)
        ctx.save_for_backward(x, w1, w2, w3, x1, x2)
        ctx.bias = [b1 is not None, b2 is not None, b3 is not None]
        return x5

    @staticmethod
    def _linear_bw(dy: torch.Tensor, x: torch.Tensor, bias: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not bias:
            return (dy.transpose(-2, -1) @ x, None)
        db = torch.empty([dy.shape[1]], dtype=dy.dtype, device=dy.device)
        dw = torch.empty([dy.shape[1], x.shape[1]], dtype=dy.dtype, device=dy.device)
        GemmFusedSumOp.OPERATOR(dy.transpose(-2, -1), x, dw, db)
        return (dw, db)

    @classmethod
    @torch.cuda.amp.custom_bwd
    def backward(cls, ctx, dx5):
        x, w1, w2, w3, x1, x2 = ctx.saved_tensors
        w1w2 = stack_or_none([w1, w2], dim=0)
        dx4 = dx5 @ w3
        dx1dx2, x4 = torch.ops.xformers.silu_bw_fused(x1, x2, dx4)
        dx1, dx2 = dx1dx2.unbind(1)
        del x1, x2, dx4
        dw3, db3 = cls._linear_bw(dx5, x4, bias=ctx.bias[2])
        del x4, dx5
        if w1w2 is not None:
            assert dx1dx2.is_contiguous()
            assert w1w2.is_contiguous()
            w1w2 = w1w2.view([w1.shape[0] * 2, w1.shape[1]])
            dx = dx1dx2.view([dx1.shape[0], 2 * dx1.shape[1]]) @ w1w2
            dw1dw2 = dx1dx2.view([dx1.shape[0], 2 * dx1.shape[1]]).transpose(-2, -1) @ x
            dw1dw2, db1db2 = cls._linear_bw(dx1dx2.view([dx1.shape[0], 2 * dx1.shape[1]]), x, bias=ctx.bias[0])
            dw1, dw2 = dw1dw2.view([2, *w1.shape]).unbind(0)
            if ctx.bias[0]:
                db1db2 = db1db2.view([2, dx1.shape[1]])
                db1, db2 = torch.unbind(db1db2, dim=0)
            else:
                db1 = db2 = None
        else:
            dx = dx2 @ w2
            torch.addmm(dx, dx1, w1.to(dx1.dtype), beta=1, alpha=1, out=dx)
            dw2, db2 = cls._linear_bw(dx2, x, bias=ctx.bias[1])
            dw1, db1 = cls._linear_bw(dx1, x, bias=ctx.bias[0])
        return (dx, dw1, db1, dw2, db2, dw3, db3)