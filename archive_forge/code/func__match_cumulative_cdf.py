import numpy as np
from .._shared import utils
def _match_cumulative_cdf(source, template):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    """
    if source.dtype.kind == 'u':
        src_lookup = source.reshape(-1)
        src_counts = np.bincount(src_lookup)
        tmpl_counts = np.bincount(template.reshape(-1))
        tmpl_values = np.nonzero(tmpl_counts)[0]
        tmpl_counts = tmpl_counts[tmpl_values]
    else:
        src_values, src_lookup, src_counts = np.unique(source.reshape(-1), return_inverse=True, return_counts=True)
        tmpl_values, tmpl_counts = np.unique(template.reshape(-1), return_counts=True)
    src_quantiles = np.cumsum(src_counts) / source.size
    tmpl_quantiles = np.cumsum(tmpl_counts) / template.size
    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[src_lookup].reshape(source.shape)