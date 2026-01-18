import warnings
from collections import namedtuple
import numpy as np
from scipy import optimize, stats
from scipy._lib._util import check_random_state
def _compute_dplus(cdfvals):
    n = cdfvals.shape[-1]
    return (np.arange(1.0, n + 1) / n - cdfvals).max(axis=-1)