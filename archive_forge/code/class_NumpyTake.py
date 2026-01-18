import torch
from functools import partial
from torch.testing import make_tensor
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.common_dtype import all_types_and
import numpy as np
class NumpyTake(torch.autograd.Function):

    @staticmethod
    def forward(x, ind, ind_inv, dim):
        device = x.device
        x = to_numpy(x)
        ind = to_numpy(ind)
        return torch.tensor(np.take_along_axis(x, ind, dim), device=device)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, ind, ind_inv, dim = inputs
        ctx.save_for_backward(ind, ind_inv)
        ctx.save_for_forward(ind, ind_inv)
        ctx.dim = dim

    @staticmethod
    def backward(ctx, grad_output):
        ind, ind_inv = ctx.saved_tensors
        result = NumpyTake.apply(grad_output, ind_inv, ind, ctx.dim)
        return (result, None, None, None)

    @staticmethod
    def vmap(info, in_dims, x, ind, ind_inv, dim):
        x_bdim, ind_bdim, ind_inv_bdim, _ = in_dims
        logical_dim = x.dim() if x_bdim is None else x_bdim - 1
        dim = dim if dim >= 0 else dim + logical_dim

        def expand_bdim(x, x_bdim):
            if x_bdim is None:
                return x.expand(info.batch_size, *x.shape)
            return x.movedim(x_bdim, 0)
        x = expand_bdim(x, x_bdim)
        ind = expand_bdim(ind, ind_bdim)
        ind_inv = expand_bdim(ind_inv, ind_inv_bdim)
        return (NumpyTake.apply(x, ind, ind_inv, dim + 1), 0)

    @staticmethod
    def jvp(ctx, x_tangent, ind_tangent, ind_inv_tangent, _):
        assert ind_tangent is None
        assert ind_inv_tangent is None
        ind, ind_inv = ctx.saved_tensors
        return NumpyTake.apply(x_tangent, ind, ind_inv, ctx.dim)