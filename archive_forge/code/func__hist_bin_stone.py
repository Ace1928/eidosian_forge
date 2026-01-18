import contextlib
import functools
import operator
import warnings
import numpy as np
from numpy.core import overrides
def _hist_bin_stone(x, range):
    """
    Histogram bin estimator based on minimizing the estimated integrated squared error (ISE).

    The number of bins is chosen by minimizing the estimated ISE against the unknown true distribution.
    The ISE is estimated using cross-validation and can be regarded as a generalization of Scott's rule.
    https://en.wikipedia.org/wiki/Histogram#Scott.27s_normal_reference_rule

    This paper by Stone appears to be the origination of this rule.
    http://digitalassets.lib.berkeley.edu/sdtr/ucb/text/34.pdf

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.
    range : (float, float)
        The lower and upper range of the bins.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    n = x.size
    ptp_x = _ptp(x)
    if n <= 1 or ptp_x == 0:
        return 0

    def jhat(nbins):
        hh = ptp_x / nbins
        p_k = np.histogram(x, bins=nbins, range=range)[0] / n
        return (2 - (n + 1) * p_k.dot(p_k)) / hh
    nbins_upper_bound = max(100, int(np.sqrt(n)))
    nbins = min(_range(1, nbins_upper_bound + 1), key=jhat)
    if nbins == nbins_upper_bound:
        warnings.warn('The number of bins estimated may be suboptimal.', RuntimeWarning, stacklevel=3)
    return ptp_x / nbins