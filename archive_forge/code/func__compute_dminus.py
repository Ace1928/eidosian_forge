import warnings
from collections import namedtuple
import numpy as np
from scipy import optimize, stats
from scipy._lib._util import check_random_state
def _compute_dminus(cdfvals, axis=-1):
    n = cdfvals.shape[-1]
    return (cdfvals - np.arange(0.0, n) / n).max(axis=-1)