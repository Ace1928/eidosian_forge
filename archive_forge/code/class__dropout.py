from typing import Optional
import torch
import triton
from torch.cuda.amp import custom_bwd, custom_fwd
from xformers.components.activations import Activation, build_activation
from xformers.triton.k_activations import get_triton_activation_index
from xformers.triton.k_dropout import k_dropout_bw, k_dropout_fw
class _dropout(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, x, p, bias, activation, trainable_bias):
        x_ = x.reshape(-1, x.shape[-1]).contiguous()
        y = torch.empty_like(x_)
        M, N = x_.shape
        assert bias is None or (bias.dtype == x.dtype and bias.shape[0] == N)
        assert p > 0.0

        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
        N_BLOCK_N = triton.cdiv(N, BLOCK_N)
        seeds = torch.randint(65536, (N_BLOCK_N,), device=x.device, dtype=torch.int32)
        bias_ptr = bias if bias is not None else x_
        k_dropout_fw[grid](y, x_, bias_ptr, seeds, y.stride(0), M, N, p, x.dtype == torch.float16, USE_BIAS=bias is not None, ACTIVATION=activation, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
        if activation is not None:
            ctx.save_for_backward(seeds, bias, x)
        else:
            ctx.save_for_backward(seeds, bias, None)
        ctx.trainable_bias = bias is not None and trainable_bias
        ctx.activation = activation
        ctx.p = p
        return y.reshape_as(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        seeds, bias, inputs = ctx.saved_tensors
        grad_out_ = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()
        grad_in = torch.empty_like(grad_out_)
        M, N = grad_out_.shape
        assert inputs is not None or ctx.activation is None
        if inputs is None:
            inputs = grad_out_
        elif inputs.ndim > 2:
            inputs = inputs.reshape(-1, N)
        N_BLOCKS_M = triton.cdiv(M, BLOCK_M)
        if ctx.trainable_bias:
            grad_bias = torch.empty((N_BLOCKS_M, N), device=grad_in.device, dtype=grad_in.dtype)
        else:
            grad_bias = grad_in

        def grid(meta):
            return (N_BLOCKS_M, triton.cdiv(N, meta['BLOCK_N']))
        k_dropout_bw[grid](grad_in, grad_bias, grad_out_, inputs, bias if bias is not None else inputs, seeds, grad_out_.stride(0), inputs.stride(0), M, N, ctx.p, grad_in.dtype == torch.float16, USE_BIAS=bias is not None, ACTIVATION=ctx.activation, TRAINABLE_BIAS=ctx.trainable_bias, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
        return (grad_in.reshape_as(grad_out), None, torch.sum(grad_bias, dim=0) if ctx.trainable_bias else None, None, None, None)