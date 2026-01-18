from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from .common import BaseOperator, get_xformers_operator, register_operator
from .unbind import stack_or_none, unbind
class _SwiGLUDecomposedFunc(torch.autograd.Function):
    """
    This is just an example implementation with all
    operations explicited. This implementation is worse
    than pytorch, because pytorch is able to fuse some operations
    (eg the linear forward ...) that are decomposed here.

    The time measurements were made on the ViT-Giant setting:
    - A100/f16
    - input: [4440, 1536]
    - hidden: [4440, 4096]
    """
    NAME = 'decomposed'
    FORCE_BW_F32 = False

    def _silu_backward(dy, x):
        sigm = 1 / (1 + torch.exp(-x.float()))
        return (dy.float() * sigm * (1 + x.float() * (1 - sigm))).to(x.dtype)

    @classmethod
    def forward(cls, ctx, x, w1, b1, w2, b2, w3, b3):
        x1 = x @ w1.transpose(-2, -1) + b1
        x2 = x @ w2.transpose(-2, -1) + b2
        x3 = F.silu(x1)
        x4 = x3 * x2
        x5 = x4 @ w3.transpose(-2, -1) + b3
        ctx.save_for_backward(x, w1, b1, w2, b2, w3, b3, x1, x2, x3, x4, x5)
        return x5

    @classmethod
    def backward(cls, ctx, dx5):
        saved_tensors = ctx.saved_tensors
        if cls.FORCE_BW_F32:
            dx5 = dx5.float()
            saved_tensors = [t.float() for t in ctx.saved_tensors]
        x, w1, b1, w2, b2, w3, b3, x1, x2, x3, x4, x5 = saved_tensors
        dx4 = dx5 @ w3
        dw3 = dx5.transpose(-2, -1) @ x4
        db3 = dx5.sum(0)
        dx3 = dx4 * x2
        dx2 = dx4 * x3
        dx1 = cls._silu_backward(dx3, x1)
        dx = dx2 @ w2
        dw2 = dx2.transpose(-2, -1) @ x
        db2 = dx2.sum(0)
        dx += dx1 @ w1
        dw1 = dx1.transpose(-2, -1) @ x
        db1 = dx1.sum(0)
        return (dx, dw1, db1, dw2, db2, dw3, db3)