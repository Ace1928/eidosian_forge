import math
import torch
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd
import triton
import triton.language as tl
class LayerNormFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, residual=None, x1=None, weight1=None, bias1=None, eps=1e-06, dropout_p=0.0, rowscale=None, prenorm=False, residual_in_fp32=False, is_rms_norm=False, return_dropout_mask=False):
        x_shape_og = x.shape
        x = x.reshape(-1, x.shape[-1])
        if x.stride(-1) != 1:
            x = x.contiguous()
        if residual is not None:
            assert residual.shape == x_shape_og
            residual = residual.reshape(-1, residual.shape[-1])
            if residual.stride(-1) != 1:
                residual = residual.contiguous()
        if x1 is not None:
            assert x1.shape == x_shape_og
            assert rowscale is None, 'rowscale is not supported with parallel LayerNorm'
            x1 = x1.reshape(-1, x1.shape[-1])
            if x1.stride(-1) != 1:
                x1 = x1.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        if weight1 is not None:
            weight1 = weight1.contiguous()
        if bias1 is not None:
            bias1 = bias1.contiguous()
        if rowscale is not None:
            rowscale = rowscale.reshape(-1).contiguous()
        residual_dtype = residual.dtype if residual is not None else torch.float32 if residual_in_fp32 else None
        y, y1, mean, rstd, residual_out, seeds, dropout_mask, dropout_mask1 = _layer_norm_fwd(x, weight, bias, eps, residual, x1, weight1, bias1, dropout_p=dropout_p, rowscale=rowscale, residual_dtype=residual_dtype, is_rms_norm=is_rms_norm, return_dropout_mask=return_dropout_mask)
        ctx.save_for_backward(residual_out, weight, bias, weight1, bias1, rowscale, seeds, mean, rstd)
        ctx.x_shape_og = x_shape_og
        ctx.eps = eps
        ctx.dropout_p = dropout_p
        ctx.is_rms_norm = is_rms_norm
        ctx.has_residual = residual is not None
        ctx.has_x1 = x1 is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x.dtype
        y = y.reshape(x_shape_og)
        y1 = y1.reshape(x_shape_og) if y1 is not None else None
        residual_out = residual_out.reshape(x_shape_og) if residual_out is not None else None
        dropout_mask = dropout_mask.reshape(x_shape_og) if dropout_mask is not None else None
        dropout_mask1 = dropout_mask1.reshape(x_shape_og) if dropout_mask1 is not None else None
        if not return_dropout_mask:
            if weight1 is None:
                return y if not prenorm else (y, residual_out)
            else:
                return (y, y1) if not prenorm else (y, y1, residual_out)
        elif weight1 is None:
            return (y, dropout_mask, dropout_mask1) if not prenorm else (y, residual_out, dropout_mask, dropout_mask1)
        else:
            return (y, y1, dropout_mask, dropout_mask1) if not prenorm else (y, y1, residual_out, dropout_mask, dropout_mask1)

    @staticmethod
    def backward(ctx, dy, *args):
        x, weight, bias, weight1, bias1, rowscale, seeds, mean, rstd = ctx.saved_tensors
        dy = dy.reshape(-1, dy.shape[-1])
        if dy.stride(-1) != 1:
            dy = dy.contiguous()
        assert dy.shape == x.shape
        if weight1 is not None:
            dy1, args = (args[0], args[1:])
            dy1 = dy1.reshape(-1, dy1.shape[-1])
            if dy1.stride(-1) != 1:
                dy1 = dy1.contiguous()
            assert dy1.shape == x.shape
        else:
            dy1 = None
        if ctx.prenorm:
            dresidual = args[0]
            dresidual = dresidual.reshape(-1, dresidual.shape[-1])
            if dresidual.stride(-1) != 1:
                dresidual = dresidual.contiguous()
            assert dresidual.shape == x.shape
        else:
            dresidual = None
        dx, dw, db, dresidual_in, dx1, dw1, db1 = _layer_norm_bwd(dy, x, weight, bias, ctx.eps, mean, rstd, dresidual, dy1, weight1, bias1, seeds, ctx.dropout_p, rowscale, ctx.has_residual, ctx.has_x1, ctx.is_rms_norm, x_dtype=ctx.x_dtype)
        return (dx.reshape(ctx.x_shape_og), dw, db, dresidual_in.reshape(ctx.x_shape_og) if ctx.has_residual else None, dx1.reshape(ctx.x_shape_og) if dx1 is not None else None, dw1, db1, None, None, None, None, None, None, None)