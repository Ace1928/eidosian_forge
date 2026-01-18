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
def _get_wilcoxon_distr(n):
    """
    Distribution of probability of the Wilcoxon ranksum statistic r_plus (sum
    of ranks of positive differences).
    Returns an array with the probabilities of all the possible ranks
    r = 0, ..., n*(n+1)/2
    """
    c = np.ones(1, dtype=np.float64)
    for k in range(1, n + 1):
        prev_c = c
        c = np.zeros(k * (k + 1) // 2 + 1, dtype=np.float64)
        m = len(prev_c)
        c[:m] = prev_c * 0.5
        c[-m:] += prev_c * 0.5
    return c