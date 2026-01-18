from functools import partial
import torch
import torch.nn as nn
from bitsandbytes.triton.dequantize_rowwise import dequantize_rowwise
from bitsandbytes.triton.int8_matmul_mixed_dequantize import (
from bitsandbytes.triton.int8_matmul_rowwise_dequantize import (
from bitsandbytes.triton.quantize_columnwise_and_transpose import (
from bitsandbytes.triton.quantize_global import (
from bitsandbytes.triton.quantize_rowwise import quantize_rowwise
from bitsandbytes.triton.triton_utils import is_triton_available
class StandardLinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias=None):
        X = input.view(-1, input.size(-1))
        ctx.save_for_backward(X, weight, bias)
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output.view(*input.size()[:-1], -1)

    @staticmethod
    def backward(ctx, grad_output_3D):
        input, weight, bias = ctx.saved_tensors
        grad_output = grad_output_3D.reshape(-1, grad_output_3D.size(-1))
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight.to(grad_output.dtype)).view(*grad_output_3D.size()[:-1], -1)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().matmul(input.to(grad_output.dtype))
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        return (grad_input, grad_weight, grad_bias)