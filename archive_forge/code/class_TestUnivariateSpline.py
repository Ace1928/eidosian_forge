import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
class TestUnivariateSpline:

    def test_linear_constant(self):
        x = [1, 2, 3]
        y = [3, 3, 3]
        lut = UnivariateSpline(x, y, k=1)
        assert_array_almost_equal(lut.get_knots(), [1, 3])
        assert_array_almost_equal(lut.get_coeffs(), [3, 3])
        assert_almost_equal(lut.get_residual(), 0.0)
        assert_array_almost_equal(lut([1, 1.5, 2]), [3, 3, 3])

    def test_preserve_shape(self):
        x = [1, 2, 3]
        y = [0, 2, 4]
        lut = UnivariateSpline(x, y, k=1)
        arg = 2
        assert_equal(shape(arg), shape(lut(arg)))
        assert_equal(shape(arg), shape(lut(arg, nu=1)))
        arg = [1.5, 2, 2.5]
        assert_equal(shape(arg), shape(lut(arg)))
        assert_equal(shape(arg), shape(lut(arg, nu=1)))

    def test_linear_1d(self):
        x = [1, 2, 3]
        y = [0, 2, 4]
        lut = UnivariateSpline(x, y, k=1)
        assert_array_almost_equal(lut.get_knots(), [1, 3])
        assert_array_almost_equal(lut.get_coeffs(), [0, 4])
        assert_almost_equal(lut.get_residual(), 0.0)
        assert_array_almost_equal(lut([1, 1.5, 2]), [0, 1, 2])

    def test_subclassing(self):

        class ZeroSpline(UnivariateSpline):

            def __call__(self, x):
                return 0 * array(x)
        sp = ZeroSpline([1, 2, 3, 4, 5], [3, 2, 3, 2, 3], k=2)
        assert_array_equal(sp([1.5, 2.5]), [0.0, 0.0])

    def test_empty_input(self):
        x = [1, 3, 5, 7, 9]
        y = [0, 4, 9, 12, 21]
        spl = UnivariateSpline(x, y, k=3)
        assert_array_equal(spl([]), array([]))

    def test_roots(self):
        x = [1, 3, 5, 7, 9]
        y = [0, 4, 9, 12, 21]
        spl = UnivariateSpline(x, y, k=3)
        assert_almost_equal(spl.roots()[0], 1.050290639101332)

    def test_roots_length(self):
        x = np.linspace(0, 50 * np.pi, 1000)
        y = np.cos(x)
        spl = UnivariateSpline(x, y, s=0)
        assert_equal(len(spl.roots()), 50)

    def test_derivatives(self):
        x = [1, 3, 5, 7, 9]
        y = [0, 4, 9, 12, 21]
        spl = UnivariateSpline(x, y, k=3)
        assert_almost_equal(spl.derivatives(3.5), [5.5152902, 1.7146577, -0.1830357, 0.3125])

    def test_derivatives_2(self):
        x = np.arange(8)
        y = x ** 3 + 2.0 * x ** 2
        tck = splrep(x, y, s=0)
        ders = spalde(3, tck)
        assert_allclose(ders, [45.0, 39.0, 22.0, 6.0], atol=1e-15)
        spl = UnivariateSpline(x, y, s=0, k=3)
        assert_allclose(spl.derivatives(3), ders, atol=1e-15)

    def test_resize_regression(self):
        """Regression test for #1375."""
        x = [-1.0, -0.65016502, -0.58856235, -0.26903553, -0.17370892, -0.10011001, 0.0, 0.10011001, 0.17370892, 0.26903553, 0.58856235, 0.65016502, 1.0]
        y = [1.0, 0.62928599, 0.5797223, 0.39965815, 0.36322694, 0.3508061, 0.35214793, 0.3508061, 0.36322694, 0.39965815, 0.5797223, 0.62928599, 1.0]
        w = [1000000000000.0, 688.875973, 489.314737, 426.864807, 607.74677, 451.341444, 317.48021, 451.341444, 607.74677, 426.864807, 489.314737, 688.875973, 1000000000000.0]
        spl = UnivariateSpline(x=x, y=y, w=w, s=None)
        desired = array([0.35100374, 0.51715855, 0.87789547, 0.98719344])
        assert_allclose(spl([0.1, 0.5, 0.9, 0.99]), desired, atol=0.0005)

    def test_out_of_range_regression(self):
        x = np.arange(5, dtype=float)
        y = x ** 3
        xp = linspace(-8, 13, 100)
        xp_zeros = xp.copy()
        xp_zeros[np.logical_or(xp_zeros < 0.0, xp_zeros > 4.0)] = 0
        xp_clip = xp.copy()
        xp_clip[xp_clip < x[0]] = x[0]
        xp_clip[xp_clip > x[-1]] = x[-1]
        for cls in [UnivariateSpline, InterpolatedUnivariateSpline]:
            spl = cls(x=x, y=y)
            for ext in [0, 'extrapolate']:
                assert_allclose(spl(xp, ext=ext), xp ** 3, atol=1e-16)
                assert_allclose(cls(x, y, ext=ext)(xp), xp ** 3, atol=1e-16)
            for ext in [1, 'zeros']:
                assert_allclose(spl(xp, ext=ext), xp_zeros ** 3, atol=1e-16)
                assert_allclose(cls(x, y, ext=ext)(xp), xp_zeros ** 3, atol=1e-16)
            for ext in [2, 'raise']:
                assert_raises(ValueError, spl, xp, **dict(ext=ext))
            for ext in [3, 'const']:
                assert_allclose(spl(xp, ext=ext), xp_clip ** 3, atol=1e-16)
                assert_allclose(cls(x, y, ext=ext)(xp), xp_clip ** 3, atol=1e-16)
        t = spl.get_knots()[3:4]
        spl = LSQUnivariateSpline(x, y, t)
        assert_allclose(spl(xp, ext=0), xp ** 3, atol=1e-16)
        assert_allclose(spl(xp, ext=1), xp_zeros ** 3, atol=1e-16)
        assert_raises(ValueError, spl, xp, **dict(ext=2))
        assert_allclose(spl(xp, ext=3), xp_clip ** 3, atol=1e-16)
        for ext in [-1, 'unknown']:
            spl = UnivariateSpline(x, y)
            assert_raises(ValueError, spl, xp, **dict(ext=ext))
            assert_raises(ValueError, UnivariateSpline, **dict(x=x, y=y, ext=ext))

    def test_lsq_fpchec(self):
        xs = np.arange(100) * 1.0
        ys = np.arange(100) * 1.0
        knots = np.linspace(0, 99, 10)
        bbox = (-1, 101)
        assert_raises(ValueError, LSQUnivariateSpline, xs, ys, knots, bbox=bbox)

    def test_derivative_and_antiderivative(self):
        x = np.linspace(0, 1, 70) ** 3
        y = np.cos(x)
        spl = UnivariateSpline(x, y, s=0)
        spl2 = spl.antiderivative(2).derivative(2)
        assert_allclose(spl(0.3), spl2(0.3))
        spl2 = spl.antiderivative(1)
        assert_allclose(spl2(0.6) - spl2(0.2), spl.integral(0.2, 0.6))

    def test_derivative_extrapolation(self):
        x_values = [1, 2, 4, 6, 8.5]
        y_values = [0.5, 0.8, 1.3, 2.5, 5]
        f = UnivariateSpline(x_values, y_values, ext='const', k=3)
        x = [-1, 0, -0.5, 9, 9.5, 10]
        assert_allclose(f.derivative()(x), 0, atol=1e-15)

    def test_integral_out_of_bounds(self):
        x = np.linspace(0.0, 1.0, 7)
        for ext in range(4):
            f = UnivariateSpline(x, x, s=0, ext=ext)
            for a, b in [(1, 1), (1, 5), (2, 5), (0, 0), (-2, 0), (-2, -1)]:
                assert_allclose(f.integral(a, b), 0, atol=1e-15)

    def test_nan(self):
        x = np.arange(10, dtype=float)
        y = x ** 3
        w = np.ones_like(x)
        spl = UnivariateSpline(x, y, check_finite=True)
        t = spl.get_knots()[3:4]
        y_end = y[-1]
        for z in [np.nan, np.inf, -np.inf]:
            y[-1] = z
            assert_raises(ValueError, UnivariateSpline, **dict(x=x, y=y, check_finite=True))
            assert_raises(ValueError, InterpolatedUnivariateSpline, **dict(x=x, y=y, check_finite=True))
            assert_raises(ValueError, LSQUnivariateSpline, **dict(x=x, y=y, t=t, check_finite=True))
            y[-1] = y_end
            w[-1] = z
            assert_raises(ValueError, UnivariateSpline, **dict(x=x, y=y, w=w, check_finite=True))
            assert_raises(ValueError, InterpolatedUnivariateSpline, **dict(x=x, y=y, w=w, check_finite=True))
            assert_raises(ValueError, LSQUnivariateSpline, **dict(x=x, y=y, t=t, w=w, check_finite=True))

    def test_strictly_increasing_x(self):
        xx = np.arange(10, dtype=float)
        yy = xx ** 3
        x = np.arange(10, dtype=float)
        x[1] = x[0]
        y = x ** 3
        w = np.ones_like(x)
        spl = UnivariateSpline(xx, yy, check_finite=True)
        t = spl.get_knots()[3:4]
        UnivariateSpline(x=x, y=y, w=w, s=1, check_finite=True)
        LSQUnivariateSpline(x=x, y=y, t=t, w=w, check_finite=True)
        assert_raises(ValueError, UnivariateSpline, **dict(x=x, y=y, s=0, check_finite=True))
        assert_raises(ValueError, InterpolatedUnivariateSpline, **dict(x=x, y=y, check_finite=True))

    def test_increasing_x(self):
        xx = np.arange(10, dtype=float)
        yy = xx ** 3
        x = np.arange(10, dtype=float)
        x[1] = x[0] - 1.0
        y = x ** 3
        w = np.ones_like(x)
        spl = UnivariateSpline(xx, yy, check_finite=True)
        t = spl.get_knots()[3:4]
        assert_raises(ValueError, UnivariateSpline, **dict(x=x, y=y, check_finite=True))
        assert_raises(ValueError, InterpolatedUnivariateSpline, **dict(x=x, y=y, check_finite=True))
        assert_raises(ValueError, LSQUnivariateSpline, **dict(x=x, y=y, t=t, w=w, check_finite=True))

    def test_invalid_input_for_univariate_spline(self):
        with assert_raises(ValueError) as info:
            x_values = [1, 2, 4, 6, 8.5]
            y_values = [0.5, 0.8, 1.3, 2.5]
            UnivariateSpline(x_values, y_values)
        assert 'x and y should have a same length' in str(info.value)
        with assert_raises(ValueError) as info:
            x_values = [1, 2, 4, 6, 8.5]
            y_values = [0.5, 0.8, 1.3, 2.5, 2.8]
            w_values = [-1.0, 1.0, 1.0, 1.0]
            UnivariateSpline(x_values, y_values, w=w_values)
        assert 'x, y, and w should have a same length' in str(info.value)
        with assert_raises(ValueError) as info:
            bbox = -1
            UnivariateSpline(x_values, y_values, bbox=bbox)
        assert 'bbox shape should be (2,)' in str(info.value)
        with assert_raises(ValueError) as info:
            UnivariateSpline(x_values, y_values, k=6)
        assert 'k should be 1 <= k <= 5' in str(info.value)
        with assert_raises(ValueError) as info:
            UnivariateSpline(x_values, y_values, s=-1.0)
        assert 's should be s >= 0.0' in str(info.value)

    def test_invalid_input_for_interpolated_univariate_spline(self):
        with assert_raises(ValueError) as info:
            x_values = [1, 2, 4, 6, 8.5]
            y_values = [0.5, 0.8, 1.3, 2.5]
            InterpolatedUnivariateSpline(x_values, y_values)
        assert 'x and y should have a same length' in str(info.value)
        with assert_raises(ValueError) as info:
            x_values = [1, 2, 4, 6, 8.5]
            y_values = [0.5, 0.8, 1.3, 2.5, 2.8]
            w_values = [-1.0, 1.0, 1.0, 1.0]
            InterpolatedUnivariateSpline(x_values, y_values, w=w_values)
        assert 'x, y, and w should have a same length' in str(info.value)
        with assert_raises(ValueError) as info:
            bbox = -1
            InterpolatedUnivariateSpline(x_values, y_values, bbox=bbox)
        assert 'bbox shape should be (2,)' in str(info.value)
        with assert_raises(ValueError) as info:
            InterpolatedUnivariateSpline(x_values, y_values, k=6)
        assert 'k should be 1 <= k <= 5' in str(info.value)

    def test_invalid_input_for_lsq_univariate_spline(self):
        x_values = [1, 2, 4, 6, 8.5]
        y_values = [0.5, 0.8, 1.3, 2.5, 2.8]
        spl = UnivariateSpline(x_values, y_values, check_finite=True)
        t_values = spl.get_knots()[3:4]
        with assert_raises(ValueError) as info:
            x_values = [1, 2, 4, 6, 8.5]
            y_values = [0.5, 0.8, 1.3, 2.5]
            LSQUnivariateSpline(x_values, y_values, t_values)
        assert 'x and y should have a same length' in str(info.value)
        with assert_raises(ValueError) as info:
            x_values = [1, 2, 4, 6, 8.5]
            y_values = [0.5, 0.8, 1.3, 2.5, 2.8]
            w_values = [1.0, 1.0, 1.0, 1.0]
            LSQUnivariateSpline(x_values, y_values, t_values, w=w_values)
        assert 'x, y, and w should have a same length' in str(info.value)
        message = 'Interior knots t must satisfy Schoenberg-Whitney conditions'
        with assert_raises(ValueError, match=message) as info:
            bbox = (100, -100)
            LSQUnivariateSpline(x_values, y_values, t_values, bbox=bbox)
        with assert_raises(ValueError) as info:
            bbox = -1
            LSQUnivariateSpline(x_values, y_values, t_values, bbox=bbox)
        assert 'bbox shape should be (2,)' in str(info.value)
        with assert_raises(ValueError) as info:
            LSQUnivariateSpline(x_values, y_values, t_values, k=6)
        assert 'k should be 1 <= k <= 5' in str(info.value)

    def test_array_like_input(self):
        x_values = np.array([1, 2, 4, 6, 8.5])
        y_values = np.array([0.5, 0.8, 1.3, 2.5, 2.8])
        w_values = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        bbox = np.array([-100, 100])
        spl1 = UnivariateSpline(x=x_values, y=y_values, w=w_values, bbox=bbox)
        spl2 = UnivariateSpline(x=x_values.tolist(), y=y_values.tolist(), w=w_values.tolist(), bbox=bbox.tolist())
        assert_allclose(spl1([0.1, 0.5, 0.9, 0.99]), spl2([0.1, 0.5, 0.9, 0.99]))

    def test_fpknot_oob_crash(self):
        x = range(109)
        y = [0.0, 0.0, 0.0, 0.0, 0.0, 10.9, 0.0, 11.0, 0.0, 0.0, 0.0, 10.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.9, 0.0, 0.0, 0.0, 11.0, 0.0, 0.0, 0.0, 10.9, 0.0, 0.0, 0.0, 10.5, 0.0, 0.0, 0.0, 10.7, 0.0, 0.0, 0.0, 11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.9, 0.0, 0.0, 10.7, 0.0, 0.0, 0.0, 10.6, 0.0, 0.0, 0.0, 10.5, 0.0, 0.0, 10.7, 0.0, 0.0, 10.5, 0.0, 0.0, 11.5, 0.0, 0.0, 0.0, 10.7, 0.0, 0.0, 10.7, 0.0, 0.0, 10.9, 0.0, 0.0, 10.8, 0.0, 0.0, 0.0, 10.7, 0.0, 0.0, 10.6, 0.0, 0.0, 0.0, 10.4, 0.0, 0.0, 10.6, 0.0, 0.0, 10.5, 0.0, 0.0, 0.0, 10.7, 0.0, 0.0, 0.0, 10.4, 0.0, 0.0, 0.0, 10.8, 0.0]
        with suppress_warnings() as sup:
            r = sup.record(UserWarning, '\nThe maximal number of iterations maxit \\(set to 20 by the program\\)\nallowed for finding a smoothing spline with fp=s has been reached: s\ntoo small.\nThere is an approximation returned but the corresponding weighted sum\nof squared residuals does not satisfy the condition abs\\(fp-s\\)/s < tol.')
            UnivariateSpline(x, y, k=1)
            assert_equal(len(r), 1)