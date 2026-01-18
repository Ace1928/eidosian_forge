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
class TestPPoly:

    def test_simple(self):
        c = np.array([[1, 4], [2, 5], [3, 6]])
        x = np.array([0, 0.5, 1])
        p = PPoly(c, x)
        assert_allclose(p(0.3), 1 * 0.3 ** 2 + 2 * 0.3 + 3)
        assert_allclose(p(0.7), 4 * (0.7 - 0.5) ** 2 + 5 * (0.7 - 0.5) + 6)

    def test_periodic(self):
        c = np.array([[1, 4], [2, 5], [3, 6]])
        x = np.array([0, 0.5, 1])
        p = PPoly(c, x, extrapolate='periodic')
        assert_allclose(p(1.3), 1 * 0.3 ** 2 + 2 * 0.3 + 3)
        assert_allclose(p(-0.3), 4 * (0.7 - 0.5) ** 2 + 5 * (0.7 - 0.5) + 6)
        assert_allclose(p(1.3, 1), 2 * 0.3 + 2)
        assert_allclose(p(-0.3, 1), 8 * (0.7 - 0.5) + 5)

    def test_read_only(self):
        c = np.array([[1, 4], [2, 5], [3, 6]])
        x = np.array([0, 0.5, 1])
        xnew = np.array([0, 0.1, 0.2])
        PPoly(c, x, extrapolate='periodic')
        for writeable in (True, False):
            x.flags.writeable = writeable
            c.flags.writeable = writeable
            f = PPoly(c, x)
            vals = f(xnew)
            assert_(np.isfinite(vals).all())

    def test_descending(self):

        def binom_matrix(power):
            n = np.arange(power + 1).reshape(-1, 1)
            k = np.arange(power + 1)
            B = binom(n, k)
            return B[::-1, ::-1]
        np.random.seed(0)
        power = 3
        for m in [10, 20, 30]:
            x = np.sort(np.random.uniform(0, 10, m + 1))
            ca = np.random.uniform(-2, 2, size=(power + 1, m))
            h = np.diff(x)
            h_powers = h[None, :] ** np.arange(power + 1)[::-1, None]
            B = binom_matrix(power)
            cap = ca * h_powers
            cdp = np.dot(B.T, cap)
            cd = cdp / h_powers
            pa = PPoly(ca, x, extrapolate=True)
            pd = PPoly(cd[:, ::-1], x[::-1], extrapolate=True)
            x_test = np.random.uniform(-10, 20, 100)
            assert_allclose(pa(x_test), pd(x_test), rtol=1e-13)
            assert_allclose(pa(x_test, 1), pd(x_test, 1), rtol=1e-13)
            pa_d = pa.derivative()
            pd_d = pd.derivative()
            assert_allclose(pa_d(x_test), pd_d(x_test), rtol=1e-13)
            pa_i = pa.antiderivative()
            pd_i = pd.antiderivative()
            for a, b in np.random.uniform(-10, 20, (5, 2)):
                int_a = pa.integrate(a, b)
                int_d = pd.integrate(a, b)
                assert_allclose(int_a, int_d, rtol=1e-13)
                assert_allclose(pa_i(b) - pa_i(a), pd_i(b) - pd_i(a), rtol=1e-13)
            roots_d = pd.roots()
            roots_a = pa.roots()
            assert_allclose(roots_a, np.sort(roots_d), rtol=1e-12)

    def test_multi_shape(self):
        c = np.random.rand(6, 2, 1, 2, 3)
        x = np.array([0, 0.5, 1])
        p = PPoly(c, x)
        assert_equal(p.x.shape, x.shape)
        assert_equal(p.c.shape, c.shape)
        assert_equal(p(0.3).shape, c.shape[2:])
        assert_equal(p(np.random.rand(5, 6)).shape, (5, 6) + c.shape[2:])
        dp = p.derivative()
        assert_equal(dp.c.shape, (5, 2, 1, 2, 3))
        ip = p.antiderivative()
        assert_equal(ip.c.shape, (7, 2, 1, 2, 3))

    def test_construct_fast(self):
        np.random.seed(1234)
        c = np.array([[1, 4], [2, 5], [3, 6]], dtype=float)
        x = np.array([0, 0.5, 1])
        p = PPoly.construct_fast(c, x)
        assert_allclose(p(0.3), 1 * 0.3 ** 2 + 2 * 0.3 + 3)
        assert_allclose(p(0.7), 4 * (0.7 - 0.5) ** 2 + 5 * (0.7 - 0.5) + 6)

    def test_vs_alternative_implementations(self):
        np.random.seed(1234)
        c = np.random.rand(3, 12, 22)
        x = np.sort(np.r_[0, np.random.rand(11), 1])
        p = PPoly(c, x)
        xp = np.r_[0.3, 0.5, 0.33, 0.6]
        expected = _ppoly_eval_1(c, x, xp)
        assert_allclose(p(xp), expected)
        expected = _ppoly_eval_2(c[:, :, 0], x, xp)
        assert_allclose(p(xp)[:, 0], expected)

    def test_from_spline(self):
        np.random.seed(1234)
        x = np.sort(np.r_[0, np.random.rand(11), 1])
        y = np.random.rand(len(x))
        spl = splrep(x, y, s=0)
        pp = PPoly.from_spline(spl)
        xi = np.linspace(0, 1, 200)
        assert_allclose(pp(xi), splev(xi, spl))
        b = BSpline(*spl)
        ppp = PPoly.from_spline(b)
        assert_allclose(ppp(xi), b(xi))
        t, c, k = spl
        for extrap in (None, True, False):
            b = BSpline(t, c, k, extrapolate=extrap)
            p = PPoly.from_spline(b)
            assert_equal(p.extrapolate, b.extrapolate)

    def test_derivative_simple(self):
        np.random.seed(1234)
        c = np.array([[4, 3, 2, 1]]).T
        dc = np.array([[3 * 4, 2 * 3, 2]]).T
        ddc = np.array([[2 * 3 * 4, 1 * 2 * 3]]).T
        x = np.array([0, 1])
        pp = PPoly(c, x)
        dpp = PPoly(dc, x)
        ddpp = PPoly(ddc, x)
        assert_allclose(pp.derivative().c, dpp.c)
        assert_allclose(pp.derivative(2).c, ddpp.c)

    def test_derivative_eval(self):
        np.random.seed(1234)
        x = np.sort(np.r_[0, np.random.rand(11), 1])
        y = np.random.rand(len(x))
        spl = splrep(x, y, s=0)
        pp = PPoly.from_spline(spl)
        xi = np.linspace(0, 1, 200)
        for dx in range(0, 3):
            assert_allclose(pp(xi, dx), splev(xi, spl, dx))

    def test_derivative(self):
        np.random.seed(1234)
        x = np.sort(np.r_[0, np.random.rand(11), 1])
        y = np.random.rand(len(x))
        spl = splrep(x, y, s=0, k=5)
        pp = PPoly.from_spline(spl)
        xi = np.linspace(0, 1, 200)
        for dx in range(0, 10):
            assert_allclose(pp(xi, dx), pp.derivative(dx)(xi), err_msg='dx=%d' % (dx,))

    def test_antiderivative_of_constant(self):
        p = PPoly([[1.0]], [0, 1])
        assert_equal(p.antiderivative().c, PPoly([[1], [0]], [0, 1]).c)
        assert_equal(p.antiderivative().x, PPoly([[1], [0]], [0, 1]).x)

    def test_antiderivative_regression_4355(self):
        p = PPoly([[1.0, 0.5]], [0, 1, 2])
        q = p.antiderivative()
        assert_equal(q.c, [[1, 0.5], [0, 1]])
        assert_equal(q.x, [0, 1, 2])
        assert_allclose(p.integrate(0, 2), 1.5)
        assert_allclose(q(2) - q(0), 1.5)

    def test_antiderivative_simple(self):
        np.random.seed(1234)
        c = np.array([[3, 2, 1], [0, 0, 1.6875]]).T
        ic = np.array([[1, 1, 1, 0], [0, 0, 1.6875, 0.328125]]).T
        iic = np.array([[1 / 4, 1 / 3, 1 / 2, 0, 0], [0, 0, 1.6875 / 2, 0.328125, 0.037434895833333336]]).T
        x = np.array([0, 0.25, 1])
        pp = PPoly(c, x)
        ipp = pp.antiderivative()
        iipp = pp.antiderivative(2)
        iipp2 = ipp.antiderivative()
        assert_allclose(ipp.x, x)
        assert_allclose(ipp.c.T, ic.T)
        assert_allclose(iipp.c.T, iic.T)
        assert_allclose(iipp2.c.T, iic.T)

    def test_antiderivative_vs_derivative(self):
        np.random.seed(1234)
        x = np.linspace(0, 1, 30) ** 2
        y = np.random.rand(len(x))
        spl = splrep(x, y, s=0, k=5)
        pp = PPoly.from_spline(spl)
        for dx in range(0, 10):
            ipp = pp.antiderivative(dx)
            pp2 = ipp.derivative(dx)
            assert_allclose(pp.c, pp2.c)
            for k in range(dx):
                pp2 = ipp.derivative(k)
                r = 1e-13
                endpoint = r * pp2.x[:-1] + (1 - r) * pp2.x[1:]
                assert_allclose(pp2(pp2.x[1:]), pp2(endpoint), rtol=1e-07, err_msg='dx=%d k=%d' % (dx, k))

    def test_antiderivative_vs_spline(self):
        np.random.seed(1234)
        x = np.sort(np.r_[0, np.random.rand(11), 1])
        y = np.random.rand(len(x))
        spl = splrep(x, y, s=0, k=5)
        pp = PPoly.from_spline(spl)
        for dx in range(0, 10):
            pp2 = pp.antiderivative(dx)
            spl2 = splantider(spl, dx)
            xi = np.linspace(0, 1, 200)
            assert_allclose(pp2(xi), splev(xi, spl2), rtol=1e-07)

    def test_antiderivative_continuity(self):
        c = np.array([[2, 1, 2, 2], [2, 1, 3, 3]]).T
        x = np.array([0, 0.5, 1])
        p = PPoly(c, x)
        ip = p.antiderivative()
        assert_allclose(ip(0.5 - 1e-09), ip(0.5 + 1e-09), rtol=1e-08)
        p2 = ip.derivative()
        assert_allclose(p2.c, p.c)

    def test_integrate(self):
        np.random.seed(1234)
        x = np.sort(np.r_[0, np.random.rand(11), 1])
        y = np.random.rand(len(x))
        spl = splrep(x, y, s=0, k=5)
        pp = PPoly.from_spline(spl)
        a, b = (0.3, 0.9)
        ig = pp.integrate(a, b)
        ipp = pp.antiderivative()
        assert_allclose(ig, ipp(b) - ipp(a))
        assert_allclose(ig, splint(a, b, spl))
        a, b = (-0.3, 0.9)
        ig = pp.integrate(a, b, extrapolate=True)
        assert_allclose(ig, ipp(b) - ipp(a))
        assert_(np.isnan(pp.integrate(a, b, extrapolate=False)).all())

    def test_integrate_readonly(self):
        x = np.array([1, 2, 4])
        c = np.array([[0.0, 0.0], [-1.0, -1.0], [2.0, -0.0], [1.0, 2.0]])
        for writeable in (True, False):
            x.flags.writeable = writeable
            P = PPoly(c, x)
            vals = P.integrate(1, 4)
            assert_(np.isfinite(vals).all())

    def test_integrate_periodic(self):
        x = np.array([1, 2, 4])
        c = np.array([[0.0, 0.0], [-1.0, -1.0], [2.0, -0.0], [1.0, 2.0]])
        P = PPoly(c, x, extrapolate='periodic')
        I = P.antiderivative()
        period_int = I(4) - I(1)
        assert_allclose(P.integrate(1, 4), period_int)
        assert_allclose(P.integrate(-10, -7), period_int)
        assert_allclose(P.integrate(-10, -4), 2 * period_int)
        assert_allclose(P.integrate(1.5, 2.5), I(2.5) - I(1.5))
        assert_allclose(P.integrate(3.5, 5), I(2) - I(1) + I(4) - I(3.5))
        assert_allclose(P.integrate(3.5 + 12, 5 + 12), I(2) - I(1) + I(4) - I(3.5))
        assert_allclose(P.integrate(3.5, 5 + 12), I(2) - I(1) + I(4) - I(3.5) + 4 * period_int)
        assert_allclose(P.integrate(0, -1), I(2) - I(3))
        assert_allclose(P.integrate(-9, -10), I(2) - I(3))
        assert_allclose(P.integrate(0, -10), I(2) - I(3) - 3 * period_int)

    def test_roots(self):
        x = np.linspace(0, 1, 31) ** 2
        y = np.sin(30 * x)
        spl = splrep(x, y, s=0, k=3)
        pp = PPoly.from_spline(spl)
        r = pp.roots()
        r = r[(r >= 0 - 1e-15) & (r <= 1 + 1e-15)]
        assert_allclose(r, sproot(spl), atol=1e-15)

    def test_roots_idzero(self):
        c = np.array([[-1, 0.25], [0, 0], [-1, 0.25]]).T
        x = np.array([0, 0.4, 0.6, 1.0])
        pp = PPoly(c, x)
        assert_array_equal(pp.roots(), [0.25, 0.4, np.nan, 0.6 + 0.25])
        const = 2.0
        c1 = c.copy()
        c1[1, :] += const
        pp1 = PPoly(c1, x)
        assert_array_equal(pp1.solve(const), [0.25, 0.4, np.nan, 0.6 + 0.25])

    def test_roots_all_zero(self):
        c = [[0], [0]]
        x = [0, 1]
        p = PPoly(c, x)
        assert_array_equal(p.roots(), [0, np.nan])
        assert_array_equal(p.solve(0), [0, np.nan])
        assert_array_equal(p.solve(1), [])
        c = [[0, 0], [0, 0]]
        x = [0, 1, 2]
        p = PPoly(c, x)
        assert_array_equal(p.roots(), [0, np.nan, 1, np.nan])
        assert_array_equal(p.solve(0), [0, np.nan, 1, np.nan])
        assert_array_equal(p.solve(1), [])

    def test_roots_repeated(self):
        c = np.array([[1, 0, -1], [-1, 0, 0]]).T
        x = np.array([-1, 0, 1])
        pp = PPoly(c, x)
        assert_array_equal(pp.roots(), [-2, 0])
        assert_array_equal(pp.roots(extrapolate=False), [0])

    def test_roots_discont(self):
        c = np.array([[1], [-1]]).T
        x = np.array([0, 0.5, 1])
        pp = PPoly(c, x)
        assert_array_equal(pp.roots(), [0.5])
        assert_array_equal(pp.roots(discontinuity=False), [])
        assert_array_equal(pp.solve(0.5), [0.5])
        assert_array_equal(pp.solve(0.5, discontinuity=False), [])
        assert_array_equal(pp.solve(1.5), [])
        assert_array_equal(pp.solve(1.5, discontinuity=False), [])

    def test_roots_random(self):
        np.random.seed(1234)
        num = 0
        for extrapolate in (True, False):
            for order in range(0, 20):
                x = np.unique(np.r_[0, 10 * np.random.rand(30), 10])
                c = 2 * np.random.rand(order + 1, len(x) - 1, 2, 3) - 1
                pp = PPoly(c, x)
                for y in [0, np.random.random()]:
                    r = pp.solve(y, discontinuity=False, extrapolate=extrapolate)
                    for i in range(2):
                        for j in range(3):
                            rr = r[i, j]
                            if rr.size > 0:
                                num += rr.size
                                val = pp(rr, extrapolate=extrapolate)[:, i, j]
                                cmpval = pp(rr, nu=1, extrapolate=extrapolate)[:, i, j]
                                msg = f'({extrapolate!r}) r = {repr(rr)}'
                                assert_allclose((val - y) / cmpval, 0, atol=1e-07, err_msg=msg)
        assert_(num > 100, repr(num))

    def test_roots_croots(self):
        np.random.seed(1234)
        for k in range(1, 15):
            c = np.random.rand(k, 1, 130)
            if k == 3:
                c[:, 0, 0] = (1, 2, 1)
            for y in [0, np.random.random()]:
                w = np.empty(c.shape, dtype=complex)
                _ppoly._croots_poly1(c, w)
                if k == 1:
                    assert_(np.isnan(w).all())
                    continue
                res = 0
                cres = 0
                for i in range(k):
                    res += c[i, None] * w ** (k - 1 - i)
                    cres += abs(c[i, None] * w ** (k - 1 - i))
                with np.errstate(invalid='ignore'):
                    res /= cres
                res = res.ravel()
                res = res[~np.isnan(res)]
                assert_allclose(res, 0, atol=1e-10)

    def test_extrapolate_attr(self):
        c = np.array([[-1, 0, 1]]).T
        x = np.array([0, 1])
        for extrapolate in [True, False, None]:
            pp = PPoly(c, x, extrapolate=extrapolate)
            pp_d = pp.derivative()
            pp_i = pp.antiderivative()
            if extrapolate is False:
                assert_(np.isnan(pp([-0.1, 1.1])).all())
                assert_(np.isnan(pp_i([-0.1, 1.1])).all())
                assert_(np.isnan(pp_d([-0.1, 1.1])).all())
                assert_equal(pp.roots(), [1])
            else:
                assert_allclose(pp([-0.1, 1.1]), [1 - 0.1 ** 2, 1 - 1.1 ** 2])
                assert_(not np.isnan(pp_i([-0.1, 1.1])).any())
                assert_(not np.isnan(pp_d([-0.1, 1.1])).any())
                assert_allclose(pp.roots(), [1, -1])