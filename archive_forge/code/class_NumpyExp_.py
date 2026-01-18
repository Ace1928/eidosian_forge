import torch
from functools import partial
from torch.testing import make_tensor
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.common_dtype import all_types_and
import numpy as np
class NumpyExp_(torch.autograd.Function):

    @staticmethod
    def forward(x):
        x_np = to_numpy(x)
        np.exp(x_np, x_np)
        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, = inputs
        ctx.mark_dirty(x)
        ctx.save_for_backward(output)
        ctx.save_for_forward(output)

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        return NumpyMul.apply(grad_output, output)

    @staticmethod
    def vmap(info, in_dims, x):
        NumpyExp_.apply(x)
        return (x, in_dims[0])

    @staticmethod
    def jvp(ctx, x_tangent):
        output, = ctx.saved_tensors
        x_tangent.mul_(output)
        return x_tangent