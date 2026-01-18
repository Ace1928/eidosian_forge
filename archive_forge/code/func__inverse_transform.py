import warnings
import sys
from functools import partial
import numpy as np
from numpy.random import RandomState
from numpy.testing import (assert_array_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import re
from scipy import optimize, stats, special
from scipy.stats._morestats import _abw_state, _get_As_weibull, _Avals_weibull
from .common_tests import check_named_results
from .._hypotests import _get_wilcoxon_distr, _get_wilcoxon_distr2
from scipy.stats._binomtest import _binary_search_for_binom_tst
from scipy.stats._distr_params import distcont
def _inverse_transform(x, lmbda):
    x_inv = np.zeros(x.shape, dtype=x.dtype)
    pos = x >= 0
    if abs(lmbda) < np.spacing(1.0):
        x_inv[pos] = np.exp(x[pos]) - 1
    else:
        x_inv[pos] = np.power(x[pos] * lmbda + 1, 1 / lmbda) - 1
    if abs(lmbda - 2) > np.spacing(1.0):
        x_inv[~pos] = 1 - np.power(-(2 - lmbda) * x[~pos] + 1, 1 / (2 - lmbda))
    else:
        x_inv[~pos] = 1 - np.exp(-x[~pos])
    return x_inv