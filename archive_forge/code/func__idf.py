import numpy as np
from numpy import float64, ndarray
import numpy.ma as ma
from numpy.ma import MaskedArray
from . import _mstats_basic as mstats
from scipy.stats.distributions import norm, beta, t, binom
def _idf(data):
    x = data.compressed()
    n = len(x)
    if n < 3:
        return [np.nan, np.nan]
    j, h = divmod(n / 4.0 + 5 / 12.0, 1)
    j = int(j)
    qlo = (1 - h) * x[j - 1] + h * x[j]
    k = n - j
    qup = (1 - h) * x[k] + h * x[k - 1]
    return [qlo, qup]