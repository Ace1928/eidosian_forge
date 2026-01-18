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
class TestBSpline:

    def test_ctor(self):
        assert_raises((TypeError, ValueError), BSpline, **dict(t=[1, 1j], c=[1.0], k=0))
        with np.errstate(invalid='ignore'):
            assert_raises(ValueError, BSpline, **dict(t=[1, np.nan], c=[1.0], k=0))
        assert_raises(ValueError, BSpline, **dict(t=[1, np.inf], c=[1.0], k=0))
        assert_raises(ValueError, BSpline, **dict(t=[1, -1], c=[1.0], k=0))
        assert_raises(ValueError, BSpline, **dict(t=[[1], [1]], c=[1.0], k=0))
        assert_raises(ValueError, BSpline, **dict(t=[0, 1, 2], c=[1], k=0))
        assert_raises(ValueError, BSpline, **dict(t=[0, 1, 2, 3, 4], c=[1.0, 1.0], k=2))
        assert_raises(TypeError, BSpline, **dict(t=[0.0, 0.0, 1.0, 2.0, 3.0, 4.0], c=[1.0, 1.0, 1.0], k='cubic'))
        assert_raises(TypeError, BSpline, **dict(t=[0.0, 0.0, 1.0, 2.0, 3.0, 4.0], c=[1.0, 1.0, 1.0], k=2.5))
        assert_raises(ValueError, BSpline, **dict(t=[0.0, 0, 1, 1, 2, 3], c=[1.0, 1, 1], k=2))
        n, k = (11, 3)
        t = np.arange(n + k + 1)
        c = np.random.random(n)
        b = BSpline(t, c, k)
        assert_allclose(t, b.t)
        assert_allclose(c, b.c)
        assert_equal(k, b.k)

    def test_tck(self):
        b = _make_random_spline()
        tck = b.tck
        assert_allclose(b.t, tck[0], atol=1e-15, rtol=1e-15)
        assert_allclose(b.c, tck[1], atol=1e-15, rtol=1e-15)
        assert_equal(b.k, tck[2])
        with pytest.raises(AttributeError):
            b.tck = 'foo'

    def test_degree_0(self):
        xx = np.linspace(0, 1, 10)
        b = BSpline(t=[0, 1], c=[3.0], k=0)
        assert_allclose(b(xx), 3)
        b = BSpline(t=[0, 0.35, 1], c=[3, 4], k=0)
        assert_allclose(b(xx), np.where(xx < 0.35, 3, 4))

    def test_degree_1(self):
        t = [0, 1, 2, 3, 4]
        c = [1, 2, 3]
        k = 1
        b = BSpline(t, c, k)
        x = np.linspace(1, 3, 50)
        assert_allclose(c[0] * B_012(x) + c[1] * B_012(x - 1) + c[2] * B_012(x - 2), b(x), atol=1e-14)
        assert_allclose(splev(x, (t, c, k)), b(x), atol=1e-14)

    def test_bernstein(self):
        k = 3
        t = np.asarray([0] * (k + 1) + [1] * (k + 1))
        c = np.asarray([1.0, 2.0, 3.0, 4.0])
        bp = BPoly(c.reshape(-1, 1), [0, 1])
        bspl = BSpline(t, c, k)
        xx = np.linspace(-1.0, 2.0, 10)
        assert_allclose(bp(xx, extrapolate=True), bspl(xx, extrapolate=True), atol=1e-14)
        assert_allclose(splev(xx, (t, c, k)), bspl(xx), atol=1e-14)

    def test_rndm_naive_eval(self):
        b = _make_random_spline()
        t, c, k = b.tck
        xx = np.linspace(t[k], t[-k - 1], 50)
        y_b = b(xx)
        y_n = [_naive_eval(x, t, c, k) for x in xx]
        assert_allclose(y_b, y_n, atol=1e-14)
        y_n2 = [_naive_eval_2(x, t, c, k) for x in xx]
        assert_allclose(y_b, y_n2, atol=1e-14)

    def test_rndm_splev(self):
        b = _make_random_spline()
        t, c, k = b.tck
        xx = np.linspace(t[k], t[-k - 1], 50)
        assert_allclose(b(xx), splev(xx, (t, c, k)), atol=1e-14)

    def test_rndm_splrep(self):
        np.random.seed(1234)
        x = np.sort(np.random.random(20))
        y = np.random.random(20)
        tck = splrep(x, y)
        b = BSpline(*tck)
        t, k = (b.t, b.k)
        xx = np.linspace(t[k], t[-k - 1], 80)
        assert_allclose(b(xx), splev(xx, tck), atol=1e-14)

    def test_rndm_unity(self):
        b = _make_random_spline()
        b.c = np.ones_like(b.c)
        xx = np.linspace(b.t[b.k], b.t[-b.k - 1], 100)
        assert_allclose(b(xx), 1.0)

    def test_vectorization(self):
        n, k = (22, 3)
        t = np.sort(np.random.random(n))
        c = np.random.random(size=(n, 6, 7))
        b = BSpline(t, c, k)
        tm, tp = (t[k], t[-k - 1])
        xx = tm + (tp - tm) * np.random.random((3, 4, 5))
        assert_equal(b(xx).shape, (3, 4, 5, 6, 7))

    def test_len_c(self):
        n, k = (33, 3)
        t = np.sort(np.random.random(n + k + 1))
        c = np.random.random(n)
        c_pad = np.r_[c, np.random.random(k + 1)]
        b, b_pad = (BSpline(t, c, k), BSpline(t, c_pad, k))
        dt = t[-1] - t[0]
        xx = np.linspace(t[0] - dt, t[-1] + dt, 50)
        assert_allclose(b(xx), b_pad(xx), atol=1e-14)
        assert_allclose(b(xx), splev(xx, (t, c, k)), atol=1e-14)
        assert_allclose(b(xx), splev(xx, (t, c_pad, k)), atol=1e-14)

    def test_endpoints(self):
        b = _make_random_spline()
        t, _, k = b.tck
        tm, tp = (t[k], t[-k - 1])
        for extrap in (True, False):
            assert_allclose(b([tm, tp], extrap), b([tm + 1e-10, tp - 1e-10], extrap), atol=1e-09)

    def test_continuity(self):
        b = _make_random_spline()
        t, _, k = b.tck
        assert_allclose(b(t[k + 1:-k - 1] - 1e-10), b(t[k + 1:-k - 1] + 1e-10), atol=1e-09)

    def test_extrap(self):
        b = _make_random_spline()
        t, c, k = b.tck
        dt = t[-1] - t[0]
        xx = np.linspace(t[k] - dt, t[-k - 1] + dt, 50)
        mask = (t[k] < xx) & (xx < t[-k - 1])
        assert_allclose(b(xx[mask], extrapolate=True), b(xx[mask], extrapolate=False))
        assert_allclose(b(xx, extrapolate=True), splev(xx, (t, c, k), ext=0))

    def test_default_extrap(self):
        b = _make_random_spline()
        t, _, k = b.tck
        xx = [t[0] - 1, t[-1] + 1]
        yy = b(xx)
        assert_(not np.all(np.isnan(yy)))

    def test_periodic_extrap(self):
        np.random.seed(1234)
        t = np.sort(np.random.random(8))
        c = np.random.random(4)
        k = 3
        b = BSpline(t, c, k, extrapolate='periodic')
        n = t.size - (k + 1)
        dt = t[-1] - t[0]
        xx = np.linspace(t[k] - dt, t[n] + dt, 50)
        xy = t[k] + (xx - t[k]) % (t[n] - t[k])
        assert_allclose(b(xx), splev(xy, (t, c, k)))
        xx = [-1, 0, 0.5, 1]
        xy = t[k] + (xx - t[k]) % (t[n] - t[k])
        assert_equal(b(xx, extrapolate='periodic'), b(xy, extrapolate=True))

    def test_ppoly(self):
        b = _make_random_spline()
        t, c, k = b.tck
        pp = PPoly.from_spline((t, c, k))
        xx = np.linspace(t[k], t[-k], 100)
        assert_allclose(b(xx), pp(xx), atol=1e-14, rtol=1e-14)

    def test_derivative_rndm(self):
        b = _make_random_spline()
        t, c, k = b.tck
        xx = np.linspace(t[0], t[-1], 50)
        xx = np.r_[xx, t]
        for der in range(1, k + 1):
            yd = splev(xx, (t, c, k), der=der)
            assert_allclose(yd, b(xx, nu=der), atol=1e-14)
        assert_allclose(b(xx, nu=k + 1), 0, atol=1e-14)

    def test_derivative_jumps(self):
        k = 2
        t = [-1, -1, 0, 1, 1, 3, 4, 6, 6, 6, 7, 7]
        np.random.seed(1234)
        c = np.r_[0, 0, np.random.random(5), 0, 0]
        b = BSpline(t, c, k)
        x = np.asarray([1, 3, 4, 6])
        assert_allclose(b(x[x != 6] - 1e-10), b(x[x != 6] + 1e-10))
        assert_(not np.allclose(b(6.0 - 1e-10), b(6 + 1e-10)))
        x0 = np.asarray([3, 4])
        assert_allclose(b(x0 - 1e-10, nu=1), b(x0 + 1e-10, nu=1))
        x1 = np.asarray([1, 6])
        assert_(not np.all(np.allclose(b(x1 - 1e-10, nu=1), b(x1 + 1e-10, nu=1))))
        assert_(not np.all(np.allclose(b(x - 1e-10, nu=2), b(x + 1e-10, nu=2))))

    def test_basis_element_quadratic(self):
        xx = np.linspace(-1, 4, 20)
        b = BSpline.basis_element(t=[0, 1, 2, 3])
        assert_allclose(b(xx), splev(xx, (b.t, b.c, b.k)), atol=1e-14)
        assert_allclose(b(xx), B_0123(xx), atol=1e-14)
        b = BSpline.basis_element(t=[0, 1, 1, 2])
        xx = np.linspace(0, 2, 10)
        assert_allclose(b(xx), np.where(xx < 1, xx * xx, (2.0 - xx) ** 2), atol=1e-14)

    def test_basis_element_rndm(self):
        b = _make_random_spline()
        t, c, k = b.tck
        xx = np.linspace(t[k], t[-k - 1], 20)
        assert_allclose(b(xx), _sum_basis_elements(xx, t, c, k), atol=1e-14)

    def test_cmplx(self):
        b = _make_random_spline()
        t, c, k = b.tck
        cc = c * (1.0 + 3j)
        b = BSpline(t, cc, k)
        b_re = BSpline(t, b.c.real, k)
        b_im = BSpline(t, b.c.imag, k)
        xx = np.linspace(t[k], t[-k - 1], 20)
        assert_allclose(b(xx).real, b_re(xx), atol=1e-14)
        assert_allclose(b(xx).imag, b_im(xx), atol=1e-14)

    def test_nan(self):
        b = BSpline.basis_element([0, 1, 1, 2])
        assert_(np.isnan(b(np.nan)))

    def test_derivative_method(self):
        b = _make_random_spline(k=5)
        t, c, k = b.tck
        b0 = BSpline(t, c, k)
        xx = np.linspace(t[k], t[-k - 1], 20)
        for j in range(1, k):
            b = b.derivative()
            assert_allclose(b0(xx, j), b(xx), atol=1e-12, rtol=1e-12)

    def test_antiderivative_method(self):
        b = _make_random_spline()
        t, c, k = b.tck
        xx = np.linspace(t[k], t[-k - 1], 20)
        assert_allclose(b.antiderivative().derivative()(xx), b(xx), atol=1e-14, rtol=1e-14)
        c = np.c_[c, c, c]
        c = np.dstack((c, c))
        b = BSpline(t, c, k)
        assert_allclose(b.antiderivative().derivative()(xx), b(xx), atol=1e-14, rtol=1e-14)

    def test_integral(self):
        b = BSpline.basis_element([0, 1, 2])
        assert_allclose(b.integrate(0, 1), 0.5)
        assert_allclose(b.integrate(1, 0), -1 * 0.5)
        assert_allclose(b.integrate(1, 0), -0.5)
        assert_allclose(b.integrate(-1, 1), 0)
        assert_allclose(b.integrate(-1, 1, extrapolate=True), 0)
        assert_allclose(b.integrate(-1, 1, extrapolate=False), 0.5)
        assert_allclose(b.integrate(1, -1, extrapolate=False), -1 * 0.5)
        assert_allclose(b.integrate(1, -1, extrapolate=False), _impl.splint(1, -1, b.tck))
        b.extrapolate = 'periodic'
        i = b.antiderivative()
        period_int = i(2) - i(0)
        assert_allclose(b.integrate(0, 2), period_int)
        assert_allclose(b.integrate(2, 0), -1 * period_int)
        assert_allclose(b.integrate(-9, -7), period_int)
        assert_allclose(b.integrate(-8, -4), 2 * period_int)
        assert_allclose(b.integrate(0.5, 1.5), i(1.5) - i(0.5))
        assert_allclose(b.integrate(1.5, 3), i(1) - i(0) + i(2) - i(1.5))
        assert_allclose(b.integrate(1.5 + 12, 3 + 12), i(1) - i(0) + i(2) - i(1.5))
        assert_allclose(b.integrate(1.5, 3 + 12), i(1) - i(0) + i(2) - i(1.5) + 6 * period_int)
        assert_allclose(b.integrate(0, -1), i(0) - i(1))
        assert_allclose(b.integrate(-9, -10), i(0) - i(1))
        assert_allclose(b.integrate(0, -9), i(1) - i(2) - 4 * period_int)

    def test_integrate_ppoly(self):
        x = [0, 1, 2, 3, 4]
        b = make_interp_spline(x, x)
        b.extrapolate = 'periodic'
        p = PPoly.from_spline(b)
        for x0, x1 in [(-5, 0.5), (0.5, 5), (-4, 13)]:
            assert_allclose(b.integrate(x0, x1), p.integrate(x0, x1))

    def test_subclassing(self):

        class B(BSpline):
            pass
        b = B.basis_element([0, 1, 2, 2])
        assert_equal(b.__class__, B)
        assert_equal(b.derivative().__class__, B)
        assert_equal(b.antiderivative().__class__, B)

    @pytest.mark.parametrize('axis', range(-4, 4))
    def test_axis(self, axis):
        n, k = (22, 3)
        t = np.linspace(0, 1, n + k + 1)
        sh = [6, 7, 8]
        pos_axis = axis % 4
        sh.insert(pos_axis, n)
        c = np.random.random(size=sh)
        b = BSpline(t, c, k, axis=axis)
        assert_equal(b.c.shape, [sh[pos_axis]] + sh[:pos_axis] + sh[pos_axis + 1:])
        xp = np.random.random((3, 4, 5))
        assert_equal(b(xp).shape, sh[:pos_axis] + list(xp.shape) + sh[pos_axis + 1:])
        for ax in [-c.ndim - 1, c.ndim]:
            assert_raises(AxisError, BSpline, **dict(t=t, c=c, k=k, axis=ax))
        for b1 in [BSpline(t, c, k, axis=axis).derivative(), BSpline(t, c, k, axis=axis).derivative(2), BSpline(t, c, k, axis=axis).antiderivative(), BSpline(t, c, k, axis=axis).antiderivative(2)]:
            assert_equal(b1.axis, b.axis)

    def test_neg_axis(self):
        k = 2
        t = [0, 1, 2, 3, 4, 5, 6]
        c = np.array([[-1, 2, 0, -1], [2, 0, -3, 1]])
        spl = BSpline(t, c, k, axis=-1)
        spl0 = BSpline(t, c[0], k)
        spl1 = BSpline(t, c[1], k)
        assert_equal(spl(2.5), [spl0(2.5), spl1(2.5)])

    def test_design_matrix_bc_types(self):
        """
        Splines with different boundary conditions are built on different
        types of vectors of knots. As far as design matrix depends only on
        vector of knots, `k` and `x` it is useful to make tests for different
        boundary conditions (and as following different vectors of knots).
        """

        def run_design_matrix_tests(n, k, bc_type):
            """
            To avoid repetition of code the following function is provided.
            """
            np.random.seed(1234)
            x = np.sort(np.random.random_sample(n) * 40 - 20)
            y = np.random.random_sample(n) * 40 - 20
            if bc_type == 'periodic':
                y[0] = y[-1]
            bspl = make_interp_spline(x, y, k=k, bc_type=bc_type)
            c = np.eye(len(bspl.t) - k - 1)
            des_matr_def = BSpline(bspl.t, c, k)(x)
            des_matr_csr = BSpline.design_matrix(x, bspl.t, k).toarray()
            assert_allclose(des_matr_csr @ bspl.c, y, atol=1e-14)
            assert_allclose(des_matr_def, des_matr_csr, atol=1e-14)
        n = 11
        k = 3
        for bc in ['clamped', 'natural']:
            run_design_matrix_tests(n, k, bc)
        for k in range(3, 8, 2):
            run_design_matrix_tests(n, k, 'not-a-knot')
        n = 5
        for k in range(2, 7):
            run_design_matrix_tests(n, k, 'periodic')

    @pytest.mark.parametrize('extrapolate', [False, True, 'periodic'])
    @pytest.mark.parametrize('degree', range(5))
    def test_design_matrix_same_as_BSpline_call(self, extrapolate, degree):
        """Test that design_matrix(x) is equivalent to BSpline(..)(x)."""
        np.random.seed(1234)
        x = np.random.random_sample(10 * (degree + 1))
        xmin, xmax = (np.amin(x), np.amax(x))
        k = degree
        t = np.r_[np.linspace(xmin - 2, xmin - 1, degree), np.linspace(xmin, xmax, 2 * (degree + 1)), np.linspace(xmax + 1, xmax + 2, degree)]
        c = np.eye(len(t) - k - 1)
        bspline = BSpline(t, c, k, extrapolate)
        assert_allclose(bspline(x), BSpline.design_matrix(x, t, k, extrapolate).toarray())
        x = np.array([xmin - 10, xmin - 1, xmax + 1.5, xmax + 10])
        if not extrapolate:
            with pytest.raises(ValueError):
                BSpline.design_matrix(x, t, k, extrapolate)
        else:
            assert_allclose(bspline(x), BSpline.design_matrix(x, t, k, extrapolate).toarray())

    def test_design_matrix_x_shapes(self):
        np.random.seed(1234)
        n = 10
        k = 3
        x = np.sort(np.random.random_sample(n) * 40 - 20)
        y = np.random.random_sample(n) * 40 - 20
        bspl = make_interp_spline(x, y, k=k)
        for i in range(1, 4):
            xc = x[:i]
            yc = y[:i]
            des_matr_csr = BSpline.design_matrix(xc, bspl.t, k).toarray()
            assert_allclose(des_matr_csr @ bspl.c, yc, atol=1e-14)

    def test_design_matrix_t_shapes(self):
        t = [1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0]
        des_matr = BSpline.design_matrix(2.0, t, 3).toarray()
        assert_allclose(des_matr, [[0.25, 0.58333333, 0.16666667, 0.0]], atol=1e-14)

    def test_design_matrix_asserts(self):
        np.random.seed(1234)
        n = 10
        k = 3
        x = np.sort(np.random.random_sample(n) * 40 - 20)
        y = np.random.random_sample(n) * 40 - 20
        bspl = make_interp_spline(x, y, k=k)
        with assert_raises(ValueError):
            BSpline.design_matrix(x, bspl.t[::-1], k)
        k = 2
        t = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        x = [1.0, 2.0, 3.0, 4.0]
        with assert_raises(ValueError):
            BSpline.design_matrix(x, t, k)

    @pytest.mark.parametrize('bc_type', ['natural', 'clamped', 'periodic', 'not-a-knot'])
    def test_from_power_basis(self, bc_type):
        np.random.seed(1234)
        x = np.sort(np.random.random(20))
        y = np.random.random(20)
        if bc_type == 'periodic':
            y[-1] = y[0]
        cb = CubicSpline(x, y, bc_type=bc_type)
        bspl = BSpline.from_power_basis(cb, bc_type=bc_type)
        xx = np.linspace(0, 1, 20)
        assert_allclose(cb(xx), bspl(xx), atol=1e-15)
        bspl_new = make_interp_spline(x, y, bc_type=bc_type)
        assert_allclose(bspl.c, bspl_new.c, atol=1e-15)

    @pytest.mark.parametrize('bc_type', ['natural', 'clamped', 'periodic', 'not-a-knot'])
    def test_from_power_basis_complex(self, bc_type):
        np.random.seed(1234)
        x = np.sort(np.random.random(20))
        y = np.random.random(20) + np.random.random(20) * 1j
        if bc_type == 'periodic':
            y[-1] = y[0]
        cb = CubicSpline(x, y, bc_type=bc_type)
        bspl = BSpline.from_power_basis(cb, bc_type=bc_type)
        bspl_new_real = make_interp_spline(x, y.real, bc_type=bc_type)
        bspl_new_imag = make_interp_spline(x, y.imag, bc_type=bc_type)
        assert_equal(bspl.c.dtype, (bspl_new_real.c + 1j * bspl_new_imag.c).dtype)
        assert_allclose(bspl.c, bspl_new_real.c + 1j * bspl_new_imag.c, atol=1e-15)

    def test_from_power_basis_exmp(self):
        """
        For x = [0, 1, 2, 3, 4] and y = [1, 1, 1, 1, 1]
        the coefficients of Cubic Spline in the power basis:

        $[[0, 0, 0, 0, 0],\\$
        $[0, 0, 0, 0, 0],\\$
        $[0, 0, 0, 0, 0],\\$
        $[1, 1, 1, 1, 1]]$

        It could be shown explicitly that coefficients of the interpolating
        function in B-spline basis are c = [1, 1, 1, 1, 1, 1, 1]
        """
        x = np.array([0, 1, 2, 3, 4])
        y = np.array([1, 1, 1, 1, 1])
        bspl = BSpline.from_power_basis(CubicSpline(x, y, bc_type='natural'), bc_type='natural')
        assert_allclose(bspl.c, [1, 1, 1, 1, 1, 1, 1], atol=1e-15)

    def test_read_only(self):
        t = np.array([0, 1])
        c = np.array([3.0])
        t.setflags(write=False)
        c.setflags(write=False)
        xx = np.linspace(0, 1, 10)
        xx.setflags(write=False)
        b = BSpline(t=t, c=c, k=0)
        assert_allclose(b(xx), 3)