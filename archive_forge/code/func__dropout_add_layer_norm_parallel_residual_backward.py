import dropout_layer_norm
import torch
from torch.nn import init
def _dropout_add_layer_norm_parallel_residual_backward(dz0, dz1, dx, x, dmask0, dmask1, mu, rsigma, gamma0, gamma1, dropout_p, has_x1, has_residual, is_rms_norm=False):
    """Assume that arguments are contiguous and aligned to 16 bytes
    dx == None means that it was a post-norm architecture
    (x = drop(x0) + residual was not returned in the fwd).
    """
    hidden_size = gamma0.numel()
    xmat = x.view((-1, hidden_size))
    dz0mat = dz0.view(xmat.shape)
    dz1mat = dz1.view(xmat.shape) if dz1 is not None else None
    dxmat = dx.view(xmat.shape) if dx is not None else None
    dx0mat, dx1mat, dresidualmat, dgamma0, dbeta0, dgamma1, dbeta1, *rest = dropout_layer_norm.dropout_add_ln_parallel_residual_bwd(dz0mat, dz1mat, dxmat, xmat, dmask0, dmask1, mu, rsigma, gamma0, gamma1, dropout_p, has_x1, has_residual, is_rms_norm)
    return (dx0mat, dx1mat, dresidualmat, dgamma0, dbeta0, dgamma1, dbeta1)