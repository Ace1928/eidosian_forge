import torch
from functools import partial
from torch.testing import make_tensor
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.common_dtype import all_types_and
import numpy as np
class CubeGenVmap(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(x):
        return (x ** 3, 3 * x ** 2)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        ctx.save_for_backward(inputs[0], outputs[1])
        ctx.save_for_forward(inputs[0], outputs[1])

    @staticmethod
    def backward(ctx, grad_output, grad_saved):
        input, dinput = ctx.saved_tensors
        result = grad_output * dinput + 6 * dinput
        return result

    @staticmethod
    def jvp(ctx, input_tangent):
        input, dinput = ctx.saved_tensors
        return (MulGenVmap.apply(input_tangent, dinput), 6 * NumpyMul.apply(input_tangent, input))