from __future__ import annotations
import warnings
import numpy as np
from itertools import combinations, permutations, product
from collections.abc import Sequence
import inspect
from scipy._lib._util import check_random_state, _rename_parameter
from scipy.special import ndtr, ndtri, comb, factorial
from scipy._lib._util import rng_integers
from dataclasses import dataclass
from ._common import ConfidenceInterval
from ._axis_nan_policy import _broadcast_concatenate, _broadcast_arrays
from ._warnings_errors import DegenerateDataWarning
def _percentile_along_axis(theta_hat_b, alpha):
    """`np.percentile` with different percentile for each slice."""
    shape = theta_hat_b.shape[:-1]
    alpha = np.broadcast_to(alpha, shape)
    percentiles = np.zeros_like(alpha, dtype=np.float64)
    for indices, alpha_i in np.ndenumerate(alpha):
        if np.isnan(alpha_i):
            msg = 'The BCa confidence interval cannot be calculated. This problem is known to occur when the distribution is degenerate or the statistic is np.min.'
            warnings.warn(DegenerateDataWarning(msg), stacklevel=3)
            percentiles[indices] = np.nan
        else:
            theta_hat_b_i = theta_hat_b[indices]
            percentiles[indices] = np.percentile(theta_hat_b_i, alpha_i)
    return percentiles[()]