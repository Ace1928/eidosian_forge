import warnings
from collections import namedtuple
import numpy as np
from scipy import optimize, stats
from scipy._lib._util import check_random_state
def _filliben(dist, data):
    X = np.sort(data, axis=-1)
    n = data.shape[-1]
    k = np.arange(1, n + 1)
    m = stats.beta(k, n + 1 - k).median()
    M = dist.ppf(m)
    return _corr(X, M)