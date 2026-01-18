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
class TestLegendreFunctions:

    def test_clpmn(self):
        z = 0.5 + 0.3j
        clp = special.clpmn(2, 2, z, 3)
        assert_array_almost_equal(clp, (array([[1.0, z, 0.5 * (3 * z * z - 1)], [0.0, sqrt(z * z - 1), 3 * z * sqrt(z * z - 1)], [0.0, 0.0, 3 * (z * z - 1)]]), array([[0.0, 1.0, 3 * z], [0.0, z / sqrt(z * z - 1), 3 * (2 * z * z - 1) / sqrt(z * z - 1)], [0.0, 0.0, 6 * z]])), 7)

    def test_clpmn_close_to_real_2(self):
        eps = 1e-10
        m = 1
        n = 3
        x = 0.5
        clp_plus = special.clpmn(m, n, x + 1j * eps, 2)[0][m, n]
        clp_minus = special.clpmn(m, n, x - 1j * eps, 2)[0][m, n]
        assert_array_almost_equal(array([clp_plus, clp_minus]), array([special.lpmv(m, n, x), special.lpmv(m, n, x)]), 7)

    def test_clpmn_close_to_real_3(self):
        eps = 1e-10
        m = 1
        n = 3
        x = 0.5
        clp_plus = special.clpmn(m, n, x + 1j * eps, 3)[0][m, n]
        clp_minus = special.clpmn(m, n, x - 1j * eps, 3)[0][m, n]
        assert_array_almost_equal(array([clp_plus, clp_minus]), array([special.lpmv(m, n, x) * np.exp(-0.5j * m * np.pi), special.lpmv(m, n, x) * np.exp(0.5j * m * np.pi)]), 7)

    def test_clpmn_across_unit_circle(self):
        eps = 1e-07
        m = 1
        n = 1
        x = 1j
        for type in [2, 3]:
            assert_almost_equal(special.clpmn(m, n, x + 1j * eps, type)[0][m, n], special.clpmn(m, n, x - 1j * eps, type)[0][m, n], 6)

    def test_inf(self):
        for z in (1, -1):
            for n in range(4):
                for m in range(1, n):
                    lp = special.clpmn(m, n, z)
                    assert_(np.isinf(lp[1][1, 1:]).all())
                    lp = special.lpmn(m, n, z)
                    assert_(np.isinf(lp[1][1, 1:]).all())

    def test_deriv_clpmn(self):
        zvals = [0.5 + 0.5j, -0.5 + 0.5j, -0.5 - 0.5j, 0.5 - 0.5j, 1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
        m = 2
        n = 3
        for type in [2, 3]:
            for z in zvals:
                for h in [0.001, 0.001j]:
                    approx_derivative = (special.clpmn(m, n, z + 0.5 * h, type)[0] - special.clpmn(m, n, z - 0.5 * h, type)[0]) / h
                    assert_allclose(special.clpmn(m, n, z, type)[1], approx_derivative, rtol=0.0001)

    def test_lpmn(self):
        lp = special.lpmn(0, 2, 0.5)
        assert_array_almost_equal(lp, (array([[1.0, 0.5, -0.125]]), array([[0.0, 1.0, 1.5]])), 4)

    def test_lpn(self):
        lpnf = special.lpn(2, 0.5)
        assert_array_almost_equal(lpnf, (array([1.0, 0.5, -0.125]), array([0.0, 1.0, 1.5])), 4)

    def test_lpmv(self):
        lp = special.lpmv(0, 2, 0.5)
        assert_almost_equal(lp, -0.125, 7)
        lp = special.lpmv(0, 40, 0.001)
        assert_almost_equal(lp, 0.1252678976534484, 7)
        with np.errstate(all='ignore'):
            lp = special.lpmv(-1, -1, 0.001)
        assert_(lp != 0 or np.isnan(lp))

    def test_lqmn(self):
        lqmnf = special.lqmn(0, 2, 0.5)
        lqf = special.lqn(2, 0.5)
        assert_array_almost_equal(lqmnf[0][0], lqf[0], 4)
        assert_array_almost_equal(lqmnf[1][0], lqf[1], 4)

    def test_lqmn_gt1(self):
        """algorithm for real arguments changes at 1.0001
           test against analytical result for m=2, n=1
        """
        x0 = 1.0001
        delta = 2e-05
        for x in (x0 - delta, x0 + delta):
            lq = special.lqmn(2, 1, x)[0][-1, -1]
            expected = 2 / (x * x - 1)
            assert_almost_equal(lq, expected)

    def test_lqmn_shape(self):
        a, b = special.lqmn(4, 4, 1.1)
        assert_equal(a.shape, (5, 5))
        assert_equal(b.shape, (5, 5))
        a, b = special.lqmn(4, 0, 1.1)
        assert_equal(a.shape, (5, 1))
        assert_equal(b.shape, (5, 1))

    def test_lqn(self):
        lqf = special.lqn(2, 0.5)
        assert_array_almost_equal(lqf, (array([0.5493, -0.7253, -0.8187]), array([1.3333, 1.216, -0.8427])), 4)