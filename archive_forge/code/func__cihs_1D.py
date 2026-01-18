import numpy as np
from numpy import float64, ndarray
import numpy.ma as ma
from numpy.ma import MaskedArray
from . import _mstats_basic as mstats
from scipy.stats.distributions import norm, beta, t, binom
def _cihs_1D(data, alpha):
    data = np.sort(data.compressed())
    n = len(data)
    alpha = min(alpha, 1 - alpha)
    k = int(binom._ppf(alpha / 2.0, n, 0.5))
    gk = binom.cdf(n - k, n, 0.5) - binom.cdf(k - 1, n, 0.5)
    if gk < 1 - alpha:
        k -= 1
        gk = binom.cdf(n - k, n, 0.5) - binom.cdf(k - 1, n, 0.5)
    gkk = binom.cdf(n - k - 1, n, 0.5) - binom.cdf(k, n, 0.5)
    I = (gk - 1 + alpha) / (gk - gkk)
    lambd = (n - k) * I / float(k + (n - 2 * k) * I)
    lims = (lambd * data[k] + (1 - lambd) * data[k - 1], lambd * data[n - k - 1] + (1 - lambd) * data[n - k])
    return lims