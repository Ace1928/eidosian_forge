import os
import operator
import itertools
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
import scipy.linalg as sl
from scipy.interpolate._bsplines import (_not_a_knot, _augknt,
import scipy.interpolate._fitpack_impl as _impl
from scipy._lib._util import AxisError
class TestInterp:
    xx = np.linspace(0.0, 2.0 * np.pi)
    yy = np.sin(xx)

    def test_non_int_order(self):
        with assert_raises(TypeError):
            make_interp_spline(self.xx, self.yy, k=2.5)

    def test_order_0(self):
        b = make_interp_spline(self.xx, self.yy, k=0)
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)
        b = make_interp_spline(self.xx, self.yy, k=0, axis=-1)
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)

    def test_linear(self):
        b = make_interp_spline(self.xx, self.yy, k=1)
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)
        b = make_interp_spline(self.xx, self.yy, k=1, axis=-1)
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)

    @pytest.mark.parametrize('k', [0, 1, 2, 3])
    def test_incompatible_x_y(self, k):
        x = [0, 1, 2, 3, 4, 5]
        y = [0, 1, 2, 3, 4, 5, 6, 7]
        with assert_raises(ValueError, match='Shapes of x'):
            make_interp_spline(x, y, k=k)

    @pytest.mark.parametrize('k', [0, 1, 2, 3])
    def test_broken_x(self, k):
        x = [0, 1, 1, 2, 3, 4]
        y = [0, 1, 2, 3, 4, 5]
        with assert_raises(ValueError, match='x to not have duplicates'):
            make_interp_spline(x, y, k=k)
        x = [0, 2, 1, 3, 4, 5]
        with assert_raises(ValueError, match='Expect x to be a 1D strictly'):
            make_interp_spline(x, y, k=k)
        x = [0, 1, 2, 3, 4, 5]
        x = np.asarray(x).reshape((1, -1))
        with assert_raises(ValueError, match='Expect x to be a 1D strictly'):
            make_interp_spline(x, y, k=k)

    def test_not_a_knot(self):
        for k in [3, 5]:
            b = make_interp_spline(self.xx, self.yy, k)
            assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)

    def test_periodic(self):
        b = make_interp_spline(self.xx, self.yy, k=5, bc_type='periodic')
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)
        for i in range(1, 5):
            assert_allclose(b(self.xx[0], nu=i), b(self.xx[-1], nu=i), atol=1e-11)
        b = make_interp_spline(self.xx, self.yy, k=5, bc_type='periodic', axis=-1)
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)
        for i in range(1, 5):
            assert_allclose(b(self.xx[0], nu=i), b(self.xx[-1], nu=i), atol=1e-11)

    @pytest.mark.parametrize('k', [2, 3, 4, 5, 6, 7])
    def test_periodic_random(self, k):
        n = 5
        np.random.seed(1234)
        x = np.sort(np.random.random_sample(n) * 10)
        y = np.random.random_sample(n) * 100
        y[0] = y[-1]
        b = make_interp_spline(x, y, k=k, bc_type='periodic')
        assert_allclose(b(x), y, atol=1e-14)

    def test_periodic_axis(self):
        n = self.xx.shape[0]
        np.random.seed(1234)
        x = np.random.random_sample(n) * 2 * np.pi
        x = np.sort(x)
        x[0] = 0.0
        x[-1] = 2 * np.pi
        y = np.zeros((2, n))
        y[0] = np.sin(x)
        y[1] = np.cos(x)
        b = make_interp_spline(x, y, k=5, bc_type='periodic', axis=1)
        for i in range(n):
            assert_allclose(b(x[i]), y[:, i], atol=1e-14)
        assert_allclose(b(x[0]), b(x[-1]), atol=1e-14)

    def test_periodic_points_exception(self):
        np.random.seed(1234)
        k = 5
        n = 8
        x = np.sort(np.random.random_sample(n))
        y = np.random.random_sample(n)
        y[0] = y[-1] - 1
        with assert_raises(ValueError):
            make_interp_spline(x, y, k=k, bc_type='periodic')

    def test_periodic_knots_exception(self):
        np.random.seed(1234)
        k = 3
        n = 7
        x = np.sort(np.random.random_sample(n))
        y = np.random.random_sample(n)
        t = np.zeros(n + 2 * k)
        with assert_raises(ValueError):
            make_interp_spline(x, y, k, t, 'periodic')

    @pytest.mark.parametrize('k', [2, 3, 4, 5])
    def test_periodic_splev(self, k):
        b = make_interp_spline(self.xx, self.yy, k=k, bc_type='periodic')
        tck = splrep(self.xx, self.yy, per=True, k=k)
        spl = splev(self.xx, tck)
        assert_allclose(spl, b(self.xx), atol=1e-14)
        for i in range(1, k):
            spl = splev(self.xx, tck, der=i)
            assert_allclose(spl, b(self.xx, nu=i), atol=1e-10)

    def test_periodic_cubic(self):
        b = make_interp_spline(self.xx, self.yy, k=3, bc_type='periodic')
        cub = CubicSpline(self.xx, self.yy, bc_type='periodic')
        assert_allclose(b(self.xx), cub(self.xx), atol=1e-14)
        n = 3
        x = np.sort(np.random.random_sample(n) * 10)
        y = np.random.random_sample(n) * 100
        y[0] = y[-1]
        b = make_interp_spline(x, y, k=3, bc_type='periodic')
        cub = CubicSpline(x, y, bc_type='periodic')
        assert_allclose(b(x), cub(x), atol=1e-14)

    def test_periodic_full_matrix(self):
        k = 3
        b = make_interp_spline(self.xx, self.yy, k=k, bc_type='periodic')
        t = _periodic_knots(self.xx, k)
        c = _make_interp_per_full_matr(self.xx, self.yy, t, k)
        b1 = np.vectorize(lambda x: _naive_eval(x, t, c, k))
        assert_allclose(b(self.xx), b1(self.xx), atol=1e-14)

    def test_quadratic_deriv(self):
        der = [(1, 8.0)]
        b = make_interp_spline(self.xx, self.yy, k=2, bc_type=(None, der))
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)
        assert_allclose(b(self.xx[-1], 1), der[0][1], atol=1e-14, rtol=1e-14)
        b = make_interp_spline(self.xx, self.yy, k=2, bc_type=(der, None))
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)
        assert_allclose(b(self.xx[0], 1), der[0][1], atol=1e-14, rtol=1e-14)

    def test_cubic_deriv(self):
        k = 3
        der_l, der_r = ([(1, 3.0)], [(1, 4.0)])
        b = make_interp_spline(self.xx, self.yy, k, bc_type=(der_l, der_r))
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)
        assert_allclose([b(self.xx[0], 1), b(self.xx[-1], 1)], [der_l[0][1], der_r[0][1]], atol=1e-14, rtol=1e-14)
        der_l, der_r = ([(2, 0)], [(2, 0)])
        b = make_interp_spline(self.xx, self.yy, k, bc_type=(der_l, der_r))
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)

    def test_quintic_derivs(self):
        k, n = (5, 7)
        x = np.arange(n).astype(np.float64)
        y = np.sin(x)
        der_l = [(1, -12.0), (2, 1)]
        der_r = [(1, 8.0), (2, 3.0)]
        b = make_interp_spline(x, y, k=k, bc_type=(der_l, der_r))
        assert_allclose(b(x), y, atol=1e-14, rtol=1e-14)
        assert_allclose([b(x[0], 1), b(x[0], 2)], [val for nu, val in der_l])
        assert_allclose([b(x[-1], 1), b(x[-1], 2)], [val for nu, val in der_r])

    @pytest.mark.xfail(reason='unstable')
    def test_cubic_deriv_unstable(self):
        k = 3
        t = _augknt(self.xx, k)
        der_l = [(1, 3.0), (2, 4.0)]
        b = make_interp_spline(self.xx, self.yy, k, t, bc_type=(der_l, None))
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)

    def test_knots_not_data_sites(self):
        k = 2
        t = np.r_[(self.xx[0],) * (k + 1), (self.xx[1:] + self.xx[:-1]) / 2.0, (self.xx[-1],) * (k + 1)]
        b = make_interp_spline(self.xx, self.yy, k, t, bc_type=([(2, 0)], [(2, 0)]))
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)
        assert_allclose([b(self.xx[0], 2), b(self.xx[-1], 2)], [0.0, 0.0], atol=1e-14)

    def test_minimum_points_and_deriv(self):
        k = 3
        x = [0.0, 1.0]
        y = [0.0, 1.0]
        b = make_interp_spline(x, y, k, bc_type=([(1, 0.0)], [(1, 3.0)]))
        xx = np.linspace(0.0, 1.0)
        yy = xx ** 3
        assert_allclose(b(xx), yy, atol=1e-14, rtol=1e-14)

    def test_deriv_spec(self):
        x = y = [1.0, 2, 3, 4, 5, 6]
        with assert_raises(ValueError):
            make_interp_spline(x, y, bc_type=([(1, 0.0)], None))
        with assert_raises(ValueError):
            make_interp_spline(x, y, bc_type=(1, 0.0))
        with assert_raises(ValueError):
            make_interp_spline(x, y, bc_type=[(1, 0.0)])
        with assert_raises(ValueError):
            make_interp_spline(x, y, bc_type=42)
        l, r = ((1, 0.0), (1, 0.0))
        with assert_raises(ValueError):
            make_interp_spline(x, y, bc_type=(l, r))

    def test_complex(self):
        k = 3
        xx = self.xx
        yy = self.yy + 1j * self.yy
        der_l, der_r = ([(1, 3j)], [(1, 4.0 + 2j)])
        b = make_interp_spline(xx, yy, k, bc_type=(der_l, der_r))
        assert_allclose(b(xx), yy, atol=1e-14, rtol=1e-14)
        assert_allclose([b(xx[0], 1), b(xx[-1], 1)], [der_l[0][1], der_r[0][1]], atol=1e-14, rtol=1e-14)
        for k in (0, 1):
            b = make_interp_spline(xx, yy, k=k)
            assert_allclose(b(xx), yy, atol=1e-14, rtol=1e-14)

    def test_int_xy(self):
        x = np.arange(10).astype(int)
        y = np.arange(10).astype(int)
        for k in (0, 1, 2, 3):
            b = make_interp_spline(x, y, k=k)
            b(x)

    def test_sliced_input(self):
        xx = np.linspace(-1, 1, 100)
        x = xx[::5]
        y = xx[::5]
        for k in (0, 1, 2, 3):
            make_interp_spline(x, y, k=k)

    def test_check_finite(self):
        x = np.arange(10).astype(float)
        y = x ** 2
        for z in [np.nan, np.inf, -np.inf]:
            y[-1] = z
            assert_raises(ValueError, make_interp_spline, x, y)

    @pytest.mark.parametrize('k', [1, 2, 3, 5])
    def test_list_input(self, k):
        x = list(range(10))
        y = [a ** 2 for a in x]
        make_interp_spline(x, y, k=k)

    def test_multiple_rhs(self):
        yy = np.c_[np.sin(self.xx), np.cos(self.xx)]
        der_l = [(1, [1.0, 2.0])]
        der_r = [(1, [3.0, 4.0])]
        b = make_interp_spline(self.xx, yy, k=3, bc_type=(der_l, der_r))
        assert_allclose(b(self.xx), yy, atol=1e-14, rtol=1e-14)
        assert_allclose(b(self.xx[0], 1), der_l[0][1], atol=1e-14, rtol=1e-14)
        assert_allclose(b(self.xx[-1], 1), der_r[0][1], atol=1e-14, rtol=1e-14)

    def test_shapes(self):
        np.random.seed(1234)
        k, n = (3, 22)
        x = np.sort(np.random.random(size=n))
        y = np.random.random(size=(n, 5, 6, 7))
        b = make_interp_spline(x, y, k)
        assert_equal(b.c.shape, (n, 5, 6, 7))
        d_l = [(1, np.random.random((5, 6, 7)))]
        d_r = [(1, np.random.random((5, 6, 7)))]
        b = make_interp_spline(x, y, k, bc_type=(d_l, d_r))
        assert_equal(b.c.shape, (n + k - 1, 5, 6, 7))

    def test_string_aliases(self):
        yy = np.sin(self.xx)
        b1 = make_interp_spline(self.xx, yy, k=3, bc_type='natural')
        b2 = make_interp_spline(self.xx, yy, k=3, bc_type=([(2, 0)], [(2, 0)]))
        assert_allclose(b1.c, b2.c, atol=1e-15)
        b1 = make_interp_spline(self.xx, yy, k=3, bc_type=('natural', 'clamped'))
        b2 = make_interp_spline(self.xx, yy, k=3, bc_type=([(2, 0)], [(1, 0)]))
        assert_allclose(b1.c, b2.c, atol=1e-15)
        b1 = make_interp_spline(self.xx, yy, k=2, bc_type=(None, 'clamped'))
        b2 = make_interp_spline(self.xx, yy, k=2, bc_type=(None, [(1, 0.0)]))
        assert_allclose(b1.c, b2.c, atol=1e-15)
        b1 = make_interp_spline(self.xx, yy, k=3, bc_type='not-a-knot')
        b2 = make_interp_spline(self.xx, yy, k=3, bc_type=None)
        assert_allclose(b1.c, b2.c, atol=1e-15)
        with assert_raises(ValueError):
            make_interp_spline(self.xx, yy, k=3, bc_type='typo')
        yy = np.c_[np.sin(self.xx), np.cos(self.xx)]
        der_l = [(1, [0.0, 0.0])]
        der_r = [(2, [0.0, 0.0])]
        b2 = make_interp_spline(self.xx, yy, k=3, bc_type=(der_l, der_r))
        b1 = make_interp_spline(self.xx, yy, k=3, bc_type=('clamped', 'natural'))
        assert_allclose(b1.c, b2.c, atol=1e-15)
        np.random.seed(1234)
        k, n = (3, 22)
        x = np.sort(np.random.random(size=n))
        y = np.random.random(size=(n, 5, 6, 7))
        d_l = [(1, np.zeros((5, 6, 7)))]
        d_r = [(1, np.zeros((5, 6, 7)))]
        b1 = make_interp_spline(x, y, k, bc_type=(d_l, d_r))
        b2 = make_interp_spline(x, y, k, bc_type='clamped')
        assert_allclose(b1.c, b2.c, atol=1e-15)

    def test_full_matrix(self):
        np.random.seed(1234)
        k, n = (3, 7)
        x = np.sort(np.random.random(size=n))
        y = np.random.random(size=n)
        t = _not_a_knot(x, k)
        b = make_interp_spline(x, y, k, t)
        cf = make_interp_full_matr(x, y, t, k)
        assert_allclose(b.c, cf, atol=1e-14, rtol=1e-14)

    def test_woodbury(self):
        """
        Random elements in diagonal matrix with blocks in the
        left lower and right upper corners checking the
        implementation of Woodbury algorithm.
        """
        np.random.seed(1234)
        n = 201
        for k in range(3, 32, 2):
            offset = int((k - 1) / 2)
            a = np.diagflat(np.random.random((1, n)))
            for i in range(1, offset + 1):
                a[:-i, i:] += np.diagflat(np.random.random((1, n - i)))
                a[i:, :-i] += np.diagflat(np.random.random((1, n - i)))
            ur = np.random.random((offset, offset))
            a[:offset, -offset:] = ur
            ll = np.random.random((offset, offset))
            a[-offset:, :offset] = ll
            d = np.zeros((k, n))
            for i, j in enumerate(range(offset, -offset - 1, -1)):
                if j < 0:
                    d[i, :j] = np.diagonal(a, offset=j)
                else:
                    d[i, j:] = np.diagonal(a, offset=j)
            b = np.random.random(n)
            assert_allclose(_woodbury_algorithm(d, ur, ll, b, k), np.linalg.solve(a, b), atol=1e-14)