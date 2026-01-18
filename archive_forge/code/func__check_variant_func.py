import functools
import itertools
import operator
import platform
import sys
import numpy as np
from numpy import (array, isnan, r_, arange, finfo, pi, sin, cos, tan, exp,
import pytest
from pytest import raises as assert_raises
from numpy.testing import (assert_equal, assert_almost_equal,
from scipy import special
import scipy.special._ufuncs as cephes
from scipy.special import ellipe, ellipk, ellipkm1
from scipy.special import elliprc, elliprd, elliprf, elliprg, elliprj
from scipy.special import mathieu_odd_coef, mathieu_even_coef, stirling2
from scipy._lib.deprecation import _NoValue
from scipy._lib._util import np_long, np_ulong
from scipy.special._basic import _FACTORIALK_LIMITS_64BITS, \
from scipy.special._testutils import with_special_errors, \
import math
def _check_variant_func(self, func, other_func, rtol, atol=0):
    np.random.seed(1234)
    n = 10000
    x = np.random.pareto(0.02, n) * (2 * np.random.randint(0, 2, n) - 1)
    y = np.random.pareto(0.02, n) * (2 * np.random.randint(0, 2, n) - 1)
    z = x + 1j * y
    with np.errstate(all='ignore'):
        w = other_func(z)
        w_real = other_func(x).real
        mask = np.isfinite(w)
        w = w[mask]
        z = z[mask]
        mask = np.isfinite(w_real)
        w_real = w_real[mask]
        x = x[mask]
        assert_func_equal(func, w, z, rtol=rtol, atol=atol)
        assert_func_equal(func, w_real, x, rtol=rtol, atol=atol)