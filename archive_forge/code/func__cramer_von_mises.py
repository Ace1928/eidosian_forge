import warnings
from collections import namedtuple
import numpy as np
from scipy import optimize, stats
from scipy._lib._util import check_random_state
def _cramer_von_mises(dist, data):
    x = np.sort(data, axis=-1)
    n = data.shape[-1]
    cdfvals = dist.cdf(x)
    u = (2 * np.arange(1, n + 1) - 1) / (2 * n)
    w = 1 / (12 * n) + np.sum((u - cdfvals) ** 2, axis=-1)
    return w