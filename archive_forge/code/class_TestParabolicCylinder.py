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
class TestParabolicCylinder:

    def test_pbdn_seq(self):
        pb = special.pbdn_seq(1, 0.1)
        assert_array_almost_equal(pb, (array([0.9975, 0.0998]), array([-0.0499, 0.9925])), 4)

    def test_pbdv(self):
        special.pbdv(1, 0.2)
        1 / 2 * 0.2 * special.pbdv(1, 0.2)[0] - special.pbdv(0, 0.2)[0]

    def test_pbdv_seq(self):
        pbn = special.pbdn_seq(1, 0.1)
        pbv = special.pbdv_seq(1, 0.1)
        assert_array_almost_equal(pbv, (real(pbn[0]), real(pbn[1])), 4)

    def test_pbdv_points(self):
        eta = np.linspace(-10, 10, 5)
        z = 2 ** (eta / 2) * np.sqrt(np.pi) / special.gamma(0.5 - 0.5 * eta)
        assert_allclose(special.pbdv(eta, 0.0)[0], z, rtol=1e-14, atol=1e-14)
        assert_allclose(special.pbdv(10.34, 20.44)[0], 1.3731383034455e-32, rtol=1e-12)
        assert_allclose(special.pbdv(-9.53, 3.44)[0], 3.166735001119246e-08, rtol=1e-12)

    def test_pbdv_gradient(self):
        x = np.linspace(-4, 4, 8)[:, None]
        eta = np.linspace(-10, 10, 5)[None, :]
        p = special.pbdv(eta, x)
        eps = 1e-07 + 1e-07 * abs(x)
        dp = (special.pbdv(eta, x + eps)[0] - special.pbdv(eta, x - eps)[0]) / eps / 2.0
        assert_allclose(p[1], dp, rtol=1e-06, atol=1e-06)

    def test_pbvv_gradient(self):
        x = np.linspace(-4, 4, 8)[:, None]
        eta = np.linspace(-10, 10, 5)[None, :]
        p = special.pbvv(eta, x)
        eps = 1e-07 + 1e-07 * abs(x)
        dp = (special.pbvv(eta, x + eps)[0] - special.pbvv(eta, x - eps)[0]) / eps / 2.0
        assert_allclose(p[1], dp, rtol=1e-06, atol=1e-06)