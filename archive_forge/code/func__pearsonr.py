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
def _pearsonr(x):
    osm_uniform = _calc_uniform_order_statistic_medians(len(x))
    xvals = distributions.norm.ppf(osm_uniform)

    def _eval_pearsonr(lmbda, xvals, samps):
        y = boxcox(samps, lmbda)
        yvals = np.sort(y)
        r, prob = _stats_py.pearsonr(xvals, yvals)
        return 1 - r
    return _optimizer(_eval_pearsonr, args=(xvals, x))