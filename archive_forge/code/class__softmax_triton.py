import logging
from typing import Optional
import torch
import triton
from torch.cuda.amp import custom_bwd, custom_fwd
from xformers.triton.k_softmax import _softmax, _softmax_backward
class _softmax_triton(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16 if _triton_softmax_fp16_enabled else None)
    def forward(ctx, x, mask, log_outputs, causal):
        """
        Fused softmax implementation, using the Triton programming model.
        This only supports a reduction over the last dimension for now
        """
        x_ = x.unsqueeze(0) if x.ndim == 2 else x
        x_ = x_.flatten(0, -3)
        if not x_.is_contiguous():
            x_ = x_.contiguous()
        y = torch.empty_like(x_)
        assert y.stride(2) == 1 and x_.stride(2) == 1, f'{x.shape} - {x_.shape} - {x_.stride()}'
        grid_2d = (x_.shape[0], x_.shape[1])
        use_mask = True
        if mask is None:
            mask = x_
            use_mask = False
        else:
            assert mask.dtype == x.dtype, 'An additive mask is requested'
        _softmax[grid_2d](y, x_, mask, y.stride(0), y.stride(1), x_.stride(0), x_.stride(1), mask.stride(0), x_.shape[2], log=log_outputs, use_mask=use_mask, causal=causal)
        ctx.save_for_backward(y)
        ctx.log_outputs = log_outputs
        ctx.causal = causal
        return y.reshape_as(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        out, = ctx.saved_tensors
        grad_out_ = grad_out.unsqueeze(0) if grad_out.ndim == 2 else grad_out
        grad_out_ = grad_out_.flatten(0, -3)
        grid_2d = (grad_out_.shape[0], grad_out_.shape[1])
        depth = triton.next_power_of_2(grad_out_.shape[2])
        grad_in = torch.empty_like(out)
        grad_in, grad_out, out = map(lambda x: x.contiguous(), [grad_in, grad_out, out])
        _softmax_backward[grid_2d](grad_in, grad_out_, out, grad_in.stride(0), grad_in.stride(1), grad_out_.stride(0), grad_out_.stride(1), out.stride(0), out.stride(1), out.shape[2], depth=depth, log=ctx.log_outputs, causal=ctx.causal)
        return (grad_in.reshape_as(grad_out), None, None, None)