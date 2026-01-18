from functools import partial
import torch
from .binary import (
from .core import is_masked_tensor, MaskedTensor, _get_data, _masks_match, _maybe_get_mask
from .passthrough import (
from .reductions import (
from .unary import (
class _MaskedWhere(torch.autograd.Function):

    @staticmethod
    def forward(ctx, cond, self, other):
        ctx.mark_non_differentiable(cond)
        ctx.save_for_backward(cond)
        return torch.ops.aten.where(cond, self, other)

    @staticmethod
    def backward(ctx, grad_output):
        cond, = ctx.saved_tensors

        def masked_out_like(mt):
            return MaskedTensor(mt.get_data(), torch.zeros_like(mt.get_mask()).bool())
        return (None, torch.ops.aten.where(cond, grad_output, masked_out_like(grad_output)), torch.ops.aten.where(cond, masked_out_like(grad_output), grad_output))