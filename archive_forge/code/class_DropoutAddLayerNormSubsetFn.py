import dropout_layer_norm
import torch
from torch.nn import init
class DropoutAddLayerNormSubsetFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x0, residual, gamma, beta, colscale, x0_subset, out_subset, dropout_p, epsilon, rowscale_const, out_numrows, residual_in_fp32=False, prenorm=False, is_rms_norm=False, return_dmask=False):
        x0 = maybe_align(x0.contiguous(), 16)
        residual = maybe_align(residual.contiguous(), 16) if residual is not None else None
        gamma = maybe_align(gamma.contiguous(), 16)
        beta = maybe_align(beta.contiguous(), 16) if beta is not None else None
        colscale = maybe_align(colscale.contiguous(), 16) if colscale is not None else None
        zmat, xmat, dmask, mu, rsigma = _dropout_add_layer_norm_subset_forward(x0, residual, gamma, beta, colscale, x0_subset, out_subset, dropout_p, epsilon, rowscale_const, out_numrows, residual_in_fp32, is_rms_norm)
        x0_saved = x0 if colscale is not None else None
        x_shape = (-1, *x0.shape[1:])
        ctx.save_for_backward(xmat.view(x_shape), x0_saved, dmask, gamma, mu, rsigma, colscale, x0_subset, out_subset)
        ctx.prenorm = prenorm
        ctx.dropout_p = dropout_p
        ctx.rowscale_const = rowscale_const
        ctx.x0_numrows = x0.shape[:-1].numel()
        ctx.has_residual = residual is not None
        ctx.is_rms_norm = is_rms_norm
        ctx.has_beta = beta is not None
        z_shape = (-1, *x0.shape[1:])
        if not return_dmask:
            return zmat.view(z_shape) if not prenorm else (zmat.view(z_shape), xmat.view(x0.shape))
        else:
            z = zmat.view(z_shape)
            dmask = dmask.view(x0.shape) if dropout_p > 0.0 else torch.ones(x0.shape, dtype=torch.uint8, device=x0.device)
            ctx.mark_non_differentiable(dmask)
            return (z, dmask) if not prenorm else (z, xmat.view(x_shape), dmask)

    @staticmethod
    def backward(ctx, dz, *args):
        dz = maybe_align(dz.contiguous(), 16)
        dx = maybe_align(args[0].contiguous(), 16) if ctx.prenorm else None
        x, x0, dmask, gamma, mu, rsigma, colscale, x0_subset, out_subset = ctx.saved_tensors
        dropout_p = ctx.dropout_p
        has_residual = ctx.has_residual
        dx0mat, dresidualmat, dgamma, dbeta, *rest = _dropout_add_layer_norm_subset_backward(dz, dx, x, x0, dmask, mu, rsigma, gamma, colscale, x0_subset, out_subset, dropout_p, ctx.rowscale_const, ctx.x0_numrows, has_residual, ctx.is_rms_norm)
        dx0 = dx0mat.view(-1, *x.shape[1:])
        dresidual = dresidualmat.view(x.shape) if dresidualmat is not None else None
        dcolscale = rest[0] if colscale is not None else None
        return (dx0, dresidual, dgamma, dbeta if ctx.has_beta else None, dcolscale, None, None, None, None, None, None, None, None, None, None)