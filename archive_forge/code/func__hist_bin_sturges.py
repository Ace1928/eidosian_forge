import contextlib
import functools
import operator
import warnings
import numpy as np
from numpy.core import overrides
def _hist_bin_sturges(x, range):
    """
    Sturges histogram bin estimator.

    A very simplistic estimator based on the assumption of normality of
    the data. This estimator has poor performance for non-normal data,
    which becomes especially obvious for large data sets. The estimate
    depends only on size of the data.

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
    return _ptp(x) / (np.log2(x.size) + 1.0)