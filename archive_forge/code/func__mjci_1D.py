import numpy as np
from numpy import float64, ndarray
import numpy.ma as ma
from numpy.ma import MaskedArray
from . import _mstats_basic as mstats
from scipy.stats.distributions import norm, beta, t, binom
def _mjci_1D(data, p):
    data = np.sort(data.compressed())
    n = data.size
    prob = (np.array(p) * n + 0.5).astype(int)
    betacdf = beta.cdf
    mj = np.empty(len(prob), float64)
    x = np.arange(1, n + 1, dtype=float64) / n
    y = x - 1.0 / n
    for i, m in enumerate(prob):
        W = betacdf(x, m - 1, n - m) - betacdf(y, m - 1, n - m)
        C1 = np.dot(W, data)
        C2 = np.dot(W, data ** 2)
        mj[i] = np.sqrt(C2 - C1 ** 2)
    return mj