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
class TestRiccati:

    def test_riccati_jn(self):
        N, x = (2, 0.2)
        S = np.empty((N, N))
        for n in range(N):
            j = special.spherical_jn(n, x)
            jp = special.spherical_jn(n, x, derivative=True)
            S[0, n] = x * j
            S[1, n] = x * jp + j
        assert_array_almost_equal(S, special.riccati_jn(n, x), 8)

    def test_riccati_yn(self):
        N, x = (2, 0.2)
        C = np.empty((N, N))
        for n in range(N):
            y = special.spherical_yn(n, x)
            yp = special.spherical_yn(n, x, derivative=True)
            C[0, n] = x * y
            C[1, n] = x * yp + y
        assert_array_almost_equal(C, special.riccati_yn(n, x), 8)