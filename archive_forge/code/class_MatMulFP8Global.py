from functools import reduce  # Required in Python 3
import operator
from typing import Optional
import warnings
import torch
from bitsandbytes.autograd._functions import GlobalOutlierPooler, MatmulLtState
import bitsandbytes.functional as F
class MatMulFP8Global(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, B, out=None, fw_code=None, bw_code=None, bsz=1024, bsz2=1024):
        ctx.is_empty = False
        if prod(A.shape) == 0:
            ctx.is_empty = True
            ctx.A = A
            ctx.B = B
            B_shape = B.shape
            if A.shape[-1] == B_shape[0]:
                return torch.empty(A.shape[:-1] + B_shape[1:], dtype=A.dtype, device=A.device)
            else:
                return torch.empty(A.shape[:-1] + B_shape[:1], dtype=A.dtype, device=A.device)
        cA, state = F.quantize(A.float(), code=fw_code)
        fp8A = F.dequantize(cA, state).to(A.dtype)
        cB, state = F.quantize(B.float(), code=fw_code)
        fp8B = F.dequantize(cB, state).to(B.dtype)
        output = torch.matmul(fp8A, fp8B)
        ctx.fw_code = fw_code
        ctx.bw_code = bw_code
        ctx.bsz = bsz
        ctx.bsz2 = bsz2
        ctx.dtype_A, ctx.dtype_B = (A.dtype, B.dtype)
        if any(ctx.needs_input_grad[:2]):
            ctx.tensors = (A, fp8B)
        else:
            ctx.tensors = (None, None)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.is_empty:
            return (torch.zeros_like(ctx.A), torch.zeros_like(ctx.B), None, None, None, None, None)
        req_gradA, req_gradB, _, _, _, _, _ = ctx.needs_input_grad
        A, B = ctx.tensors
        grad_A, grad_B = (None, None)
        cgrad_out, state = F.quantize(grad_output.float(), code=ctx.bw_code)
        fp8out = F.dequantize(cgrad_out, state).to(grad_output.dtype)
        if req_gradA:
            grad_A = torch.matmul(fp8out, B.t().to(fp8out.dtype)).to(A.dtype)
        if req_gradB:
            if len(A.shape) == 3:
                At = A.transpose(2, 1).contiguous()
            else:
                At = A.transpose(1, 0).contiguous()
            cA, state = F.quantize(At.float(), code=ctx.fw_code)
            fp8At = F.dequantize(cA, state).to(A.dtype)
            grad_B = torch.matmul(fp8At.to(fp8out.dtype), fp8out).to(B.dtype)
        return (grad_A, grad_B, None, None, None, None, None)