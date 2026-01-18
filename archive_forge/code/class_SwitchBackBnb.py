from functools import reduce  # Required in Python 3
import operator
from typing import Optional
import warnings
import torch
from bitsandbytes.autograd._functions import GlobalOutlierPooler, MatmulLtState
import bitsandbytes.functional as F
class SwitchBackBnb(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, B, out=None, bias=None, state=MatmulLtState()):
        ctx.is_empty = False
        if prod(A.shape) == 0:
            ctx.is_empty = True
            ctx.A = A
            ctx.B = B
            ctx.bias = bias
            if A.shape[-1] == B.shape[0]:
                return torch.empty(A.shape[:-1] + B.shape[1:], dtype=A.dtype, device=A.device)
            else:
                return torch.empty(A.shape[:-1] + B.shape[:1], dtype=A.dtype, device=A.device)
        formatB = state.formatB
        input_shape = A.shape
        if state.outlier_pool is None:
            state.outlier_pool = GlobalOutlierPooler.get_instance()
        if A.dtype != torch.float16:
            warnings.warn(f'MatMul8bitLt: inputs will be cast from {A.dtype} to float16 during quantization')
        if len(A.shape) == 3:
            A = A.view(-1, A.shape[-1]).contiguous()
        CA, CAt, SCA, SCAt, coo_tensorA = F.double_quant(A.to(torch.float16), threshold=state.threshold)
        if state.threshold > 0.0 and coo_tensorA is not None:
            if state.has_fp16_weights:
                idx = torch.unique(coo_tensorA.colidx).long()
                CA[:, idx] = 0
                CAt[:, idx] = 0
                subA = A[:, idx]
                state.subB = B[:, idx].t().contiguous()
                state.idx = idx
            elif state.CxB is None:
                state.CxB, state.SB = F.transform(state.CB, to_order=formatB)
        else:
            if not state.has_fp16_weights and state.CxB is None:
                state.CxB, state.SB = F.transform(state.CB, to_order=formatB)
            subA = None
        if state.has_fp16_weights:
            has_grad = True if getattr(B, 'grad', None) is not None else False
            is_transposed = not B.is_contiguous() and B.shape[0] == B.stride(1)
            if is_transposed:
                B = B.contiguous()
            if state.is_training and (not has_grad) or state.CxB is None:
                state.reset_grads()
                CB, state.CBt, state.SCB, state.SCBt, coo_tensorB = F.double_quant(B.to(torch.float16))
                state.CxB, state.SB = F.transform(CB, to_order=formatB)
        else:
            has_grad = False
        if coo_tensorA is not None and (not state.has_fp16_weights):
            outlier_idx = torch.unique(coo_tensorA.colidx)
            state.idx = outlier_idx
            outliers = F.extract_outliers(state.CxB, state.SB, state.idx.int())
            state.subB = (outliers * state.SCB.view(-1, 1) / 127.0).t().contiguous().to(A.dtype)
            CA[:, state.idx.long()] = 0
            CAt[:, state.idx.long()] = 0
            subA = A[:, state.idx.long()]
        shapeB = state.SB[0]
        if len(input_shape) == 3:
            output_shape = (input_shape[0], input_shape[1], shapeB[0])
        else:
            output_shape = (input_shape[0], shapeB[0])
        C32A, SA = F.transform(CA, 'col32')
        out32, Sout32 = F.igemmlt(C32A, state.CxB, SA, state.SB)
        if bias is None or bias.dtype == torch.float16:
            output = F.mm_dequant(out32, Sout32, SCA, state.SCB, bias=bias)
            output = output.to(A.dtype)
        else:
            output = F.mm_dequant(out32, Sout32, SCA, state.SCB, bias=None)
            output = output.to(A.dtype).add_(bias)
        if coo_tensorA is not None and subA is not None:
            output += torch.matmul(subA, state.subB)
        ctx.state = state
        ctx.formatB = formatB
        ctx.grad_shape = input_shape
        ctx.dtype_A, ctx.dtype_B, ctx.dtype_bias = (A.dtype, B.dtype, None if bias is None else bias.dtype)
        if any(ctx.needs_input_grad[:2]):
            ctx.tensors = (CAt, subA, A)
            ctx.tensor_states = (SCAt, state.idx)
        else:
            ctx.tensors = [None, None, None]
            ctx.tensor_states = (None, None)
            ctx.save_for_backward(None, None)
        clone_func = torch.clone if len(output_shape) == 3 else lambda x: x
        return clone_func(output.view(output_shape))

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.is_empty:
            bias_grad = None if ctx.bias is None else torch.zeros_like(ctx.bias)
            return (torch.zeros_like(ctx.A), torch.zeros_like(ctx.B), None, bias_grad, None)
        req_gradA, req_gradB, _, req_gradBias, _ = ctx.needs_input_grad
        CAt, subA, A = ctx.tensors
        SCAt, idx = ctx.tensor_states
        formatB = ctx.formatB
        state = ctx.state
        grad_A = grad_B = grad_bias = None
        if req_gradBias:
            grad_bias = grad_output.sum(0, dtype=ctx.dtype_bias)
        if len(grad_output.shape) == 3:
            grad_output = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
        Cgrad, Cgradt, SCgrad, SCgradt, coo_tensor = F.double_quant(grad_output.to(torch.float16))
        if req_gradB:
            grad_B = torch.matmul(grad_output.t(), A)
        if req_gradA:
            if state.CBt is not None:
                C32grad, Sgrad = F.transform(Cgrad, 'col32')
                if state.CxBt is None:
                    state.CxBt, state.SBt = F.transform(state.CBt, to_order=formatB, transpose=True)
                gradA32, SgradA32 = F.igemmlt(C32grad, state.CxBt, Sgrad, state.SBt)
                grad_A = F.mm_dequant(gradA32, SgradA32, SCgrad, state.SCBt).view(ctx.grad_shape).to(ctx.dtype_A)
            elif state.CB is not None:
                CB = state.CB.to(ctx.dtype_A, copy=True).mul_(state.SCB.unsqueeze(1).mul(1.0 / 127.0))
                grad_A = torch.matmul(grad_output, CB).view(ctx.grad_shape).to(ctx.dtype_A)
            else:
                raise Exception('State must contain either CBt or CB matrix for backward')
        return (grad_A, grad_B, None, grad_bias, None)