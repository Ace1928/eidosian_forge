from typing import List, Optional, Tuple
import torch
from torch import Tensor
from ..copy import Context as CopyContext
from ..copy import Copy
from ..phony import get_phony
from ..stream import AbstractStream, get_device
class PortalCopy(torch.autograd.Function):
    """Copies the hidden tensor in a :class:`Portal`. It replaces the hidden
    tensor with copied one.
    """

    @staticmethod
    def forward(ctx: Context, portal: Portal, prev_stream: AbstractStream, next_stream: AbstractStream, phony: Tensor) -> Tensor:
        ctx.portal = portal
        assert portal.tensor is not None
        portal.tensor, = Copy.forward(ctx, prev_stream, next_stream, portal.tensor)
        phony = get_phony(get_device(next_stream), requires_grad=False)
        return phony.detach()

    @staticmethod
    def backward(ctx: Context, grad_phony: Tensor) -> Tuple[None, None, None, None]:
        portal = ctx.portal
        assert portal.grad is not None
        _, _, portal.grad = Copy.backward(ctx, portal.grad)
        return (None, None, None, None)