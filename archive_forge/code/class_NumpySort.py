import torch
from functools import partial
from torch.testing import make_tensor
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.common_dtype import all_types_and
import numpy as np
class NumpySort(torch.autograd.Function):

    @staticmethod
    def forward(x, dim):
        device = x.device
        x = to_numpy(x)
        ind = np.argsort(x, axis=dim)
        ind_inv = np.argsort(ind, axis=dim)
        result = np.take_along_axis(x, ind, axis=dim)
        return (torch.tensor(x, device=device), torch.tensor(ind, device=device), torch.tensor(ind_inv, device=device))

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, dim = inputs
        _, ind, ind_inv = output
        ctx.mark_non_differentiable(ind, ind_inv)
        ctx.save_for_backward(ind, ind_inv)
        ctx.save_for_forward(ind, ind_inv)
        ctx.dim = dim

    @staticmethod
    def backward(ctx, grad_output, _0, _1):
        ind, ind_inv = ctx.saved_tensors
        return (NumpyTake.apply(grad_output, ind_inv, ind, ctx.dim), None)

    @staticmethod
    def vmap(info, in_dims, x, dim):
        x_bdim, _ = in_dims
        x = x.movedim(x_bdim, 0)
        dim = dim if dim >= 0 else dim + x.dim() - 1
        return (NumpySort.apply(x, dim + 1), (0, 0, 0))

    @staticmethod
    def jvp(ctx, x_tangent, _):
        ind, ind_inv = ctx.saved_tensors
        return (NumpyTake.apply(x_tangent, ind, ind_inv, ctx.dim), None, None)