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
def _poisson_means_test_iv(k1, n1, k2, n2, diff, alternative):
    if k1 != int(k1) or k2 != int(k2):
        raise TypeError('`k1` and `k2` must be integers.')
    count_err = '`k1` and `k2` must be greater than or equal to 0.'
    if k1 < 0 or k2 < 0:
        raise ValueError(count_err)
    if n1 <= 0 or n2 <= 0:
        raise ValueError('`n1` and `n2` must be greater than 0.')
    if diff < 0:
        raise ValueError('diff must be greater than or equal to 0.')
    alternatives = {'two-sided', 'less', 'greater'}
    if alternative.lower() not in alternatives:
        raise ValueError(f"Alternative must be one of '{alternatives}'.")