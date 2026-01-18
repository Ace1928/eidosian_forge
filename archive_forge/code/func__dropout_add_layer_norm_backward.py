import dropout_layer_norm
import torch
from torch.nn import init
def _dropout_add_layer_norm_backward(dz, dx, x, x0, dmask, mu, rsigma, gamma, rowscale, colscale, dropout_p, has_residual, is_rms_norm=False):
    """Assume that arguments are contiguous and aligned to 16 bytes
    dx == None means that it was a post-norm architecture
    (x = drop(x0) + residual was not returned in the fwd).
    x0 must not be None if we have colscale.
    """
    hidden_size = gamma.numel()
    xmat = x.view((-1, hidden_size))
    dzmat = dz.view(xmat.shape)
    dxmat = dx.view(xmat.shape) if dx is not None else None
    x0mat = x0.view((-1, hidden_size)) if x0 is not None else None
    rowscale = rowscale.view(-1) if rowscale is not None else None
    if colscale is not None:
        assert x0 is not None, 'x0 is required to compute the gradient of colscale'
    dx0mat, dresidualmat, dgamma, dbeta, _, _, *rest = dropout_layer_norm.dropout_add_ln_bwd(dzmat, dxmat, xmat, x0mat, dmask, mu, rsigma, gamma, rowscale, colscale, None, None, dropout_p, 1.0, 0, has_residual, is_rms_norm)
    if colscale is None:
        return (dx0mat, dresidualmat, dgamma, dbeta)
    else:
        dcolscale = rest[0]
        return (dx0mat, dresidualmat, dgamma, dbeta, dcolscale)