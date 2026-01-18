from collections import namedtuple
from dataclasses import dataclass
from math import comb
import numpy as np
import warnings
from itertools import combinations
import scipy.stats
from scipy.optimize import shgo
from . import distributions
from ._common import ConfidenceInterval
from ._continuous_distns import chi2, norm
from scipy.special import gamma, kv, gammaln
from scipy.fft import ifft
from ._stats_pythran import _a_ij_Aij_Dij2
from ._stats_pythran import (
from ._axis_nan_policy import _axis_nan_policy_factory
from scipy.stats import _stats_py
def _cdf_cvm(x, n=None):
    """
    Calculate the cdf of the Cramér-von Mises statistic for a finite sample
    size n. If N is None, use the asymptotic cdf (n=inf).

    See equation 1.8 in Csörgő, S. and Faraway, J. (1996) for finite samples,
    1.2 for the asymptotic cdf.

    The function is not expected to be accurate for large values of x, say
    x > 2, when the cdf is very close to 1 and it might return values > 1
    in that case, e.g. _cdf_cvm(2.0, 12) = 1.0000027556716846. Moreover, it
    is not accurate for small values of n, especially close to the bounds of
    the distribution's domain, [1/(12*n), n/3], where the value jumps to 0
    and 1, respectively. These are limitations of the approximation by Csörgő
    and Faraway (1996) implemented in this function.
    """
    x = np.asarray(x)
    if n is None:
        y = _cdf_cvm_inf(x)
    else:
        y = np.zeros_like(x, dtype='float')
        sup = (1.0 / (12 * n) < x) & (x < n / 3.0)
        y[sup] = _cdf_cvm_inf(x[sup]) * (1 + 1.0 / (12 * n)) + _psi1_mod(x[sup]) / n
        y[x >= n / 3] = 1
    if y.ndim == 0:
        return y[()]
    return y