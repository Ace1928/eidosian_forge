import dropout_layer_norm
import torch
from torch.nn import init
class DropoutAddLayerNormParallelResidualFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x0, x1, residual, gamma0, beta0, gamma1, beta1, dropout_p, epsilon, residual_in_fp32=False, prenorm=False, is_rms_norm=False, return_dmask=False):
        x0 = maybe_align(x0.contiguous(), 16)
        x1 = maybe_align(x1.contiguous(), 16) if x1 is not None else None
        residual = maybe_align(residual.contiguous(), 16) if residual is not None else None
        gamma0 = maybe_align(gamma0.contiguous(), 16)
        beta0 = maybe_align(beta0.contiguous(), 16) if beta0 is not None else None
        gamma1 = maybe_align(gamma1.contiguous(), 16) if gamma1 is not None else None
        beta1 = maybe_align(beta1.contiguous(), 16) if beta1 is not None else None
        z0mat, z1mat, xmat, dmask0, dmask1, mu, rsigma = _dropout_add_layer_norm_parallel_residual_forward(x0, x1, residual, gamma0, beta0, gamma1, beta1, dropout_p, epsilon, residual_in_fp32, is_rms_norm)
        ctx.save_for_backward(xmat.view(x0.shape), dmask0, dmask1, gamma0, gamma1, mu, rsigma)
        ctx.prenorm = prenorm
        ctx.dropout_p = dropout_p
        ctx.has_x1 = x1 is not None
        ctx.has_residual = residual is not None
        ctx.is_rms_norm = is_rms_norm
        ctx.has_beta = beta0 is not None
        z = (z0mat.view(x0.shape), z1mat.view(x0.shape) if z1mat is not None else None)
        if not return_dmask:
            return z if not prenorm else (*z, xmat.view(x0.shape))
        else:
            dmask0 = dmask0.view(x0.shape) if dropout_p > 0.0 else torch.ones(x0.shape, dtype=torch.uint8, device=x0.device)
            dmask1 = dmask1.view(x0.shape) if dropout_p > 0.0 and x1 is not None else torch.ones(x0.shape, dtype=torch.uint8, device=x0.device)
            ctx.mark_non_differentiable(dmask0)
            ctx.mark_non_differentiable(dmask1)
            return (*z, dmask0, dmask1) if not prenorm else (*z, xmat.view(x0.shape), dmask0, dmask1)

    @staticmethod
    def backward(ctx, dz0, dz1, *args):
        dz0 = maybe_align(dz0.contiguous(), 16)
        dz1 = maybe_align(dz1.contiguous(), 16) if dz1 is not None else None
        dx = maybe_align(args[0].contiguous(), 16) if ctx.prenorm else None
        x, dmask0, dmask1, gamma0, gamma1, mu, rsigma = ctx.saved_tensors
        dropout_p = ctx.dropout_p
        has_x1 = ctx.has_x1
        has_residual = ctx.has_residual
        dx0mat, dx1mat, dresidualmat, dgamma0, dbeta0, dgamma1, dbeta1 = _dropout_add_layer_norm_parallel_residual_backward(dz0, dz1, dx, x, dmask0, dmask1, mu, rsigma, gamma0, gamma1, dropout_p, has_x1, has_residual, ctx.is_rms_norm)
        dx0 = dx0mat.view(x.shape)
        dx1 = dx1mat.view(x.shape) if dx1mat is not None else None
        dresidual = dresidualmat.view(x.shape) if dresidualmat is not None else None
        return (dx0, dx1, dresidual, dgamma0, dbeta0 if ctx.has_beta else None, dgamma1, dbeta1 if ctx.has_beta else None, None, None, None, None, None, None)