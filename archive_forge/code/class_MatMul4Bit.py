from dataclasses import dataclass
from functools import reduce  # Required in Python 3
import operator
from typing import Callable, Optional, Tuple
import warnings
from warnings import warn
import torch
import bitsandbytes.functional as F
class MatMul4Bit(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, B, out=None, bias=None, quant_state: Optional[F.QuantState]=None):
        ctx.is_empty = False
        if prod(A.shape) == 0:
            ctx.is_empty = True
            ctx.A = A
            ctx.B = B
            ctx.bias = bias
            B_shape = quant_state.shape
            if A.shape[-1] == B_shape[0]:
                return torch.empty(A.shape[:-1] + B_shape[1:], dtype=A.dtype, device=A.device)
            else:
                return torch.empty(A.shape[:-1] + B_shape[:1], dtype=A.dtype, device=A.device)
        output = torch.nn.functional.linear(A, F.dequantize_4bit(B, quant_state).to(A.dtype).t(), bias)
        ctx.state = quant_state
        ctx.dtype_A, ctx.dtype_B, ctx.dtype_bias = (A.dtype, B.dtype, None if bias is None else bias.dtype)
        if any(ctx.needs_input_grad[:2]):
            ctx.tensors = (A, B)
        else:
            ctx.tensors = (None, None)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.is_empty:
            bias_grad = None if ctx.bias is None else torch.zeros_like(ctx.bias)
            return (torch.zeros_like(ctx.A), torch.zeros_like(ctx.B), None, bias_grad, None)
        req_gradA, _, _, req_gradBias, _ = ctx.needs_input_grad
        A, B = ctx.tensors
        grad_A, grad_B, grad_bias = (None, None, None)
        if req_gradBias:
            grad_bias = grad_output.sum(0, dtype=ctx.dtype_bias)
        if req_gradA:
            grad_A = torch.matmul(grad_output, F.dequantize_4bit(B, ctx.state).to(grad_output.dtype).t())
        return (grad_A, grad_B, None, grad_bias, None)