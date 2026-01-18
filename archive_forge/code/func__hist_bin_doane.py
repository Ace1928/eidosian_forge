import contextlib
import functools
import operator
import warnings
import numpy as np
from numpy.core import overrides
def _hist_bin_doane(x, range):
    """
    Doane's histogram bin estimator.

    Improved version of Sturges' formula which works better for
    non-normal data. See
    stats.stackexchange.com/questions/55134/doanes-formula-for-histogram-binning

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    del range
    if x.size > 2:
        sg1 = np.sqrt(6.0 * (x.size - 2) / ((x.size + 1.0) * (x.size + 3)))
        sigma = np.std(x)
        if sigma > 0.0:
            temp = x - np.mean(x)
            np.true_divide(temp, sigma, temp)
            np.power(temp, 3, temp)
            g1 = np.mean(temp)
            return _ptp(x) / (1.0 + np.log2(x.size) + np.log2(1.0 + np.absolute(g1) / sg1))
    return 0.0