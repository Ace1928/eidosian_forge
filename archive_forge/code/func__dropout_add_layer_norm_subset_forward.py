import dropout_layer_norm
import torch
from torch.nn import init
def _dropout_add_layer_norm_subset_forward(x0, residual, gamma, beta, colscale, x0_subset, out_subset, dropout_p, epsilon, rowscale_const, out_numrows, residual_in_fp32=False, is_rms_norm=False):
    """Assume that arguments are contiguous and aligned to 16 bytes"""
    hidden_size = gamma.numel()
    x0mat = x0.view((-1, hidden_size))
    residualmat = residual.view((-1, hidden_size)) if residual is not None else None
    x0_subset = x0_subset.view(-1) if x0_subset is not None else None
    out_subset = out_subset.view(-1) if out_subset is not None else None
    zmat, xmat, dmask, mu, rsigma = dropout_layer_norm.dropout_add_ln_fwd(x0mat, residualmat, gamma, beta, None, colscale, x0_subset, out_subset, dropout_p, epsilon, rowscale_const, out_numrows, None, residual_in_fp32, is_rms_norm)
    return (zmat, xmat if xmat is not None else x0mat, dmask, mu, rsigma)