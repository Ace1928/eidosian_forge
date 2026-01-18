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
def _psi1_mod(x):
    """
    psi1 is defined in equation 1.10 in Csörgő, S. and Faraway, J. (1996).
    This implements a modified version by excluding the term V(x) / 12
    (here: _cdf_cvm_inf(x) / 12) to avoid evaluating _cdf_cvm_inf(x)
    twice in _cdf_cvm.

    Implementation based on MAPLE code of Julian Faraway and R code of the
    function pCvM in the package goftest (v1.1.1), permission granted
    by Adrian Baddeley. Main difference in the implementation: the code
    here keeps adding terms of the series until the terms are small enough.
    """

    def _ed2(y):
        z = y ** 2 / 4
        b = kv(1 / 4, z) + kv(3 / 4, z)
        return np.exp(-z) * (y / 2) ** (3 / 2) * b / np.sqrt(np.pi)

    def _ed3(y):
        z = y ** 2 / 4
        c = np.exp(-z) / np.sqrt(np.pi)
        return c * (y / 2) ** (5 / 2) * (2 * kv(1 / 4, z) + 3 * kv(3 / 4, z) - kv(5 / 4, z))

    def _Ak(k, x):
        m = 2 * k + 1
        sx = 2 * np.sqrt(x)
        y1 = x ** (3 / 4)
        y2 = x ** (5 / 4)
        e1 = m * gamma(k + 1 / 2) * _ed2((4 * k + 3) / sx) / (9 * y1)
        e2 = gamma(k + 1 / 2) * _ed3((4 * k + 1) / sx) / (72 * y2)
        e3 = 2 * (m + 2) * gamma(k + 3 / 2) * _ed3((4 * k + 5) / sx) / (12 * y2)
        e4 = 7 * m * gamma(k + 1 / 2) * _ed2((4 * k + 1) / sx) / (144 * y1)
        e5 = 7 * m * gamma(k + 1 / 2) * _ed2((4 * k + 5) / sx) / (144 * y1)
        return e1 + e2 + e3 + e4 + e5
    x = np.asarray(x)
    tot = np.zeros_like(x, dtype='float')
    cond = np.ones_like(x, dtype='bool')
    k = 0
    while np.any(cond):
        z = -_Ak(k, x[cond]) / (np.pi * gamma(k + 1))
        tot[cond] = tot[cond] + z
        cond[cond] = np.abs(z) >= 1e-07
        k += 1
    return tot