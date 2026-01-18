import math
import torch
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd
import triton
import triton.language as tl
class LayerNormLinearFn(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, x, norm_weight, norm_bias, linear_weight, linear_bias, residual=None, eps=1e-06, prenorm=False, residual_in_fp32=False, is_rms_norm=False):
        x_shape_og = x.shape
        x = x.reshape(-1, x.shape[-1])
        if x.stride(-1) != 1:
            x = x.contiguous()
        if residual is not None:
            assert residual.shape == x_shape_og
            residual = residual.reshape(-1, residual.shape[-1])
            if residual.stride(-1) != 1:
                residual = residual.contiguous()
        norm_weight = norm_weight.contiguous()
        if norm_bias is not None:
            norm_bias = norm_bias.contiguous()
        residual_dtype = residual.dtype if residual is not None else torch.float32 if residual_in_fp32 else None
        y, mean, rstd, residual_out = _layer_norm_fwd(x, norm_weight, norm_bias, eps, residual, out_dtype=None if not torch.is_autocast_enabled() else torch.get_autocast_gpu_dtype(), residual_dtype=residual_dtype, is_rms_norm=is_rms_norm)
        y = y.reshape(x_shape_og)
        dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else y.dtype
        linear_weight = linear_weight.to(dtype)
        linear_bias = linear_bias.to(dtype) if linear_bias is not None else None
        out = F.linear(y.to(linear_weight.dtype), linear_weight, linear_bias)
        ctx.save_for_backward(residual_out, norm_weight, norm_bias, linear_weight, mean, rstd)
        ctx.x_shape_og = x_shape_og
        ctx.eps = eps
        ctx.is_rms_norm = is_rms_norm
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x.dtype
        ctx.linear_bias_is_none = linear_bias is None
        return out if not prenorm else (out, residual_out.reshape(x_shape_og))

    @staticmethod
    @custom_bwd
    def backward(ctx, dout, *args):
        x, norm_weight, norm_bias, linear_weight, mean, rstd = ctx.saved_tensors
        dout = dout.reshape(-1, dout.shape[-1])
        dy = F.linear(dout, linear_weight.t())
        dlinear_bias = None if ctx.linear_bias_is_none else dout.sum(0)
        if dy.stride(-1) != 1:
            dy = dy.contiguous()
        assert dy.shape == x.shape
        if ctx.prenorm:
            dresidual = args[0]
            dresidual = dresidual.reshape(-1, dresidual.shape[-1])
            if dresidual.stride(-1) != 1:
                dresidual = dresidual.contiguous()
            assert dresidual.shape == x.shape
        else:
            dresidual = None
        dx, dnorm_weight, dnorm_bias, dresidual_in, y = _layer_norm_bwd(dy, x, norm_weight, norm_bias, ctx.eps, mean, rstd, dresidual, ctx.has_residual, ctx.is_rms_norm, x_dtype=ctx.x_dtype, recompute_output=True)
        dlinear_weight = torch.einsum('bo,bi->oi', dout, y)
        return (dx.reshape(ctx.x_shape_og), dnorm_weight, dnorm_bias, dlinear_weight, dlinear_bias, dresidual_in.reshape(ctx.x_shape_og) if ctx.has_residual else None, None, None, None, None)