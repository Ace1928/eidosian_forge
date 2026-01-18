from numpy.testing import (assert_, assert_equal, assert_almost_equal,
from pytest import raises as assert_raises
import pytest
from numpy import mgrid, pi, sin, ogrid, poly1d, linspace
import numpy as np
from scipy.interpolate import (interp1d, interp2d, lagrange, PPoly, BPoly,
from scipy.special import poch, gamma
from scipy.interpolate import _ppoly
from scipy._lib._gcutils import assert_deallocated, IS_PYPY
from scipy.integrate import nquad
from scipy.special import binom
class TestBPolyFromDerivatives:

    def test_make_poly_1(self):
        c1 = BPoly._construct_from_derivatives(0, 1, [2], [3])
        assert_allclose(c1, [2.0, 3.0])

    def test_make_poly_2(self):
        c1 = BPoly._construct_from_derivatives(0, 1, [1, 0], [1])
        assert_allclose(c1, [1.0, 1.0, 1.0])
        c2 = BPoly._construct_from_derivatives(0, 1, [2, 3], [1])
        assert_allclose(c2, [2.0, 7.0 / 2, 1.0])
        c3 = BPoly._construct_from_derivatives(0, 1, [2], [1, 3])
        assert_allclose(c3, [2.0, -0.5, 1.0])

    def test_make_poly_3(self):
        c1 = BPoly._construct_from_derivatives(0, 1, [1, 2, 3], [4])
        assert_allclose(c1, [1.0, 5.0 / 3, 17.0 / 6, 4.0])
        c2 = BPoly._construct_from_derivatives(0, 1, [1], [4, 2, 3])
        assert_allclose(c2, [1.0, 19.0 / 6, 10.0 / 3, 4.0])
        c3 = BPoly._construct_from_derivatives(0, 1, [1, 2], [4, 3])
        assert_allclose(c3, [1.0, 5.0 / 3, 3.0, 4.0])

    def test_make_poly_12(self):
        np.random.seed(12345)
        ya = np.r_[0, np.random.random(5)]
        yb = np.r_[0, np.random.random(5)]
        c = BPoly._construct_from_derivatives(0, 1, ya, yb)
        pp = BPoly(c[:, None], [0, 1])
        for j in range(6):
            assert_allclose([pp(0.0), pp(1.0)], [ya[j], yb[j]])
            pp = pp.derivative()

    def test_raise_degree(self):
        np.random.seed(12345)
        x = [0, 1]
        k, d = (8, 5)
        c = np.random.random((k, 1, 2, 3, 4))
        bp = BPoly(c, x)
        c1 = BPoly._raise_degree(c, d)
        bp1 = BPoly(c1, x)
        xp = np.linspace(0, 1, 11)
        assert_allclose(bp(xp), bp1(xp))

    def test_xi_yi(self):
        assert_raises(ValueError, BPoly.from_derivatives, [0, 1], [0])

    def test_coords_order(self):
        xi = [0, 0, 1]
        yi = [[0], [0], [0]]
        assert_raises(ValueError, BPoly.from_derivatives, xi, yi)

    def test_zeros(self):
        xi = [0, 1, 2, 3]
        yi = [[0, 0], [0], [0, 0], [0, 0]]
        pp = BPoly.from_derivatives(xi, yi)
        assert_(pp.c.shape == (4, 3))
        ppd = pp.derivative()
        for xp in [0.0, 0.1, 1.0, 1.1, 1.9, 2.0, 2.5]:
            assert_allclose([pp(xp), ppd(xp)], [0.0, 0.0])

    def _make_random_mk(self, m, k):
        np.random.seed(1234)
        xi = np.asarray([1.0 * j ** 2 for j in range(m + 1)])
        yi = [np.random.random(k) for j in range(m + 1)]
        return (xi, yi)

    def test_random_12(self):
        m, k = (5, 12)
        xi, yi = self._make_random_mk(m, k)
        pp = BPoly.from_derivatives(xi, yi)
        for order in range(k // 2):
            assert_allclose(pp(xi), [yy[order] for yy in yi])
            pp = pp.derivative()

    def test_order_zero(self):
        m, k = (5, 12)
        xi, yi = self._make_random_mk(m, k)
        assert_raises(ValueError, BPoly.from_derivatives, **dict(xi=xi, yi=yi, orders=0))

    def test_orders_too_high(self):
        m, k = (5, 12)
        xi, yi = self._make_random_mk(m, k)
        BPoly.from_derivatives(xi, yi, orders=2 * k - 1)
        assert_raises(ValueError, BPoly.from_derivatives, **dict(xi=xi, yi=yi, orders=2 * k))

    def test_orders_global(self):
        m, k = (5, 12)
        xi, yi = self._make_random_mk(m, k)
        order = 5
        pp = BPoly.from_derivatives(xi, yi, orders=order)
        for j in range(order // 2 + 1):
            assert_allclose(pp(xi[1:-1] - 1e-12), pp(xi[1:-1] + 1e-12))
            pp = pp.derivative()
        assert_(not np.allclose(pp(xi[1:-1] - 1e-12), pp(xi[1:-1] + 1e-12)))
        order = 6
        pp = BPoly.from_derivatives(xi, yi, orders=order)
        for j in range(order // 2):
            assert_allclose(pp(xi[1:-1] - 1e-12), pp(xi[1:-1] + 1e-12))
            pp = pp.derivative()
        assert_(not np.allclose(pp(xi[1:-1] - 1e-12), pp(xi[1:-1] + 1e-12)))

    def test_orders_local(self):
        m, k = (7, 12)
        xi, yi = self._make_random_mk(m, k)
        orders = [o + 1 for o in range(m)]
        for i, x in enumerate(xi[1:-1]):
            pp = BPoly.from_derivatives(xi, yi, orders=orders)
            for j in range(orders[i] // 2 + 1):
                assert_allclose(pp(x - 1e-12), pp(x + 1e-12))
                pp = pp.derivative()
            assert_(not np.allclose(pp(x - 1e-12), pp(x + 1e-12)))

    def test_yi_trailing_dims(self):
        m, k = (7, 5)
        xi = np.sort(np.random.random(m + 1))
        yi = np.random.random((m + 1, k, 6, 7, 8))
        pp = BPoly.from_derivatives(xi, yi)
        assert_equal(pp.c.shape, (2 * k, m, 6, 7, 8))

    def test_gh_5430(self):
        orders = np.int32(1)
        p = BPoly.from_derivatives([0, 1], [[0], [0]], orders=orders)
        assert_almost_equal(p(0), 0)
        orders = np.int64(1)
        p = BPoly.from_derivatives([0, 1], [[0], [0]], orders=orders)
        assert_almost_equal(p(0), 0)
        orders = 1
        p = BPoly.from_derivatives([0, 1], [[0], [0]], orders=orders)
        assert_almost_equal(p(0), 0)
        orders = 1