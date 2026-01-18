import torch
from functools import partial
from torch.testing import make_tensor
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.common_dtype import all_types_and
import numpy as np
class ScaleGradGenVmap(torch.autograd.Function):
    generate_vmap_rule = True
    scale = 3.14

    @staticmethod
    def forward(x):
        return x.clone()

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ScaleGradGenVmap.scale

    @staticmethod
    def jvp(ctx, x_tangent):
        return x_tangent * ScaleGradGenVmap.scale