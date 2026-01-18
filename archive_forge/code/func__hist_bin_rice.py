import contextlib
import functools
import operator
import warnings
import numpy as np
from numpy.core import overrides
def _hist_bin_rice(x, range):
    """
    Rice histogram bin estimator.

    Another simple estimator with no normality assumption. It has better
    performance for large data than Sturges, but tends to overestimate
    the number of bins. The number of bins is proportional to the cube
    root of data size (asymptotically optimal). The estimate depends
    only on size of the data.

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
    return _ptp(x) / (2.0 * x.size ** (1.0 / 3))