from __future__ import annotations
import math
import warnings
from collections import namedtuple
import numpy as np
from numpy import (isscalar, r_, log, around, unique, asarray, zeros,
from scipy import optimize, special, interpolate, stats
from scipy._lib._bunch import _make_tuple_bunch
from scipy._lib._util import _rename_parameter, _contains_nan, _get_nan
from ._ansari_swilk_statistics import gscale, swilk
from . import _stats_py
from ._fit import FitResult
from ._stats_py import find_repeats, _normtest_finish, SignificanceResult
from .contingency import chi2_contingency
from . import distributions
from ._distn_infrastructure import rv_generic
from ._hypotests import _get_wilcoxon_distr
from ._axis_nan_policy import _axis_nan_policy_factory
@_axis_nan_policy_factory(lambda x1: (x1,), n_samples=4, n_outputs=1)
def _mood_inner_lc(xy, x, diffs, sorted_xy, n, m, N) -> float:
    diffs_prep = np.concatenate(([1], diffs))
    uniques = sorted_xy[diffs_prep != 0]
    t = np.bincount(np.cumsum(np.asarray(diffs_prep != 0, dtype=int)))[1:]
    k = len(uniques)
    js = np.arange(1, k + 1, dtype=int)
    sorted_xyx = np.sort(np.concatenate((xy, x)))
    diffs = np.diff(sorted_xyx)
    diffs_prep = np.concatenate(([1], diffs))
    diff_is_zero = np.asarray(diffs_prep != 0, dtype=int)
    xyx_counts = np.bincount(np.cumsum(diff_is_zero))[1:]
    a = xyx_counts - t
    t = np.concatenate(([0], t))
    a = np.concatenate(([0], a))
    S = np.cumsum(t)
    S_i_m1 = np.concatenate(([0], S[:-1]))

    def psi(indicator):
        return (indicator - (N + 1) / 2) ** 2
    s_lower = S[js - 1] + 1
    s_upper = S[js] + 1
    phi_J = [np.arange(s_lower[idx], s_upper[idx]) for idx in range(k)]
    phis = [np.sum(psi(I_j)) for I_j in phi_J] / t[js]
    T = sum(phis * a[js])
    E_0_T = n * (N * N - 1) / 12
    varM = m * n * (N + 1.0) * (N ** 2 - 4) / 180 - m * n / (180 * N * (N - 1)) * np.sum(t * (t ** 2 - 1) * (t ** 2 - 4 + 15 * (N - S - S_i_m1) ** 2))
    return ((T - E_0_T) / np.sqrt(varM),)