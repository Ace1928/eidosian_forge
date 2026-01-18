import torch
from functools import partial
from torch.testing import make_tensor
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.common_dtype import all_types_and
import numpy as np
class SelectGenVmap(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(x, idx):
        return x[idx]

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        x, idx = inputs
        ctx.x_shape = x.shape
        ctx.idx = idx

    @staticmethod
    def backward(ctx, grad_output):
        result = grad_output.new_zeros(ctx.x_shape)
        result[ctx.idx] = grad_output
        return (result, None)

    @staticmethod
    def jvp(ctx, x_tangent, _):
        return SelectGenVmap.apply(x_tangent, ctx.idx)