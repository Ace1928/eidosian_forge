import warnings
from collections import namedtuple
import numpy as np
from scipy import optimize, stats
from scipy._lib._util import check_random_state
def _kolmogorov_smirnov(dist, data):
    x = np.sort(data, axis=-1)
    cdfvals = dist.cdf(x)
    Dplus = _compute_dplus(cdfvals)
    Dminus = _compute_dminus(cdfvals)
    return np.maximum(Dplus, Dminus)