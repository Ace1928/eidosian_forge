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
def _somers_d(A, alternative='two-sided'):
    """Calculate Somers' D and p-value from contingency table."""
    if A.shape[0] <= 1 or A.shape[1] <= 1:
        return (np.nan, np.nan)
    NA = A.sum()
    NA2 = NA ** 2
    PA = _P(A)
    QA = _Q(A)
    Sri2 = (A.sum(axis=1) ** 2).sum()
    d = (PA - QA) / (NA2 - Sri2)
    S = _a_ij_Aij_Dij2(A) - (PA - QA) ** 2 / NA
    with np.errstate(divide='ignore'):
        Z = (PA - QA) / (4 * S) ** 0.5
    _, p = scipy.stats._stats_py._normtest_finish(Z, alternative)
    return (d, p)