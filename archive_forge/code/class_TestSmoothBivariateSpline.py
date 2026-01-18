import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
class TestSmoothBivariateSpline:

    def test_linear_constant(self):
        x = [1, 1, 1, 2, 2, 2, 3, 3, 3]
        y = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        z = [3, 3, 3, 3, 3, 3, 3, 3, 3]
        lut = SmoothBivariateSpline(x, y, z, kx=1, ky=1)
        assert_array_almost_equal(lut.get_knots(), ([1, 1, 3, 3], [1, 1, 3, 3]))
        assert_array_almost_equal(lut.get_coeffs(), [3, 3, 3, 3])
        assert_almost_equal(lut.get_residual(), 0.0)
        assert_array_almost_equal(lut([1, 1.5, 2], [1, 1.5]), [[3, 3], [3, 3], [3, 3]])

    def test_linear_1d(self):
        x = [1, 1, 1, 2, 2, 2, 3, 3, 3]
        y = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        z = [0, 0, 0, 2, 2, 2, 4, 4, 4]
        lut = SmoothBivariateSpline(x, y, z, kx=1, ky=1)
        assert_array_almost_equal(lut.get_knots(), ([1, 1, 3, 3], [1, 1, 3, 3]))
        assert_array_almost_equal(lut.get_coeffs(), [0, 0, 4, 4])
        assert_almost_equal(lut.get_residual(), 0.0)
        assert_array_almost_equal(lut([1, 1.5, 2], [1, 1.5]), [[0, 0], [1, 1], [2, 2]])

    def test_integral(self):
        x = [1, 1, 1, 2, 2, 2, 4, 4, 4]
        y = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        z = array([0, 7, 8, 3, 4, 7, 1, 3, 4])
        with suppress_warnings() as sup:
            sup.filter(UserWarning, '\nThe required storage space')
            lut = SmoothBivariateSpline(x, y, z, kx=1, ky=1, s=0)
        tx = [1, 2, 4]
        ty = [1, 2, 3]
        tz = lut(tx, ty)
        trpz = 0.25 * (diff(tx)[:, None] * diff(ty)[None, :] * (tz[:-1, :-1] + tz[1:, :-1] + tz[:-1, 1:] + tz[1:, 1:])).sum()
        assert_almost_equal(lut.integral(tx[0], tx[-1], ty[0], ty[-1]), trpz)
        lut2 = SmoothBivariateSpline(x, y, z, kx=2, ky=2, s=0)
        assert_almost_equal(lut2.integral(tx[0], tx[-1], ty[0], ty[-1]), trpz, decimal=0)
        tz = lut(tx[:-1], ty[:-1])
        trpz = 0.25 * (diff(tx[:-1])[:, None] * diff(ty[:-1])[None, :] * (tz[:-1, :-1] + tz[1:, :-1] + tz[:-1, 1:] + tz[1:, 1:])).sum()
        assert_almost_equal(lut.integral(tx[0], tx[-2], ty[0], ty[-2]), trpz)

    def test_rerun_lwrk2_too_small(self):
        x = np.linspace(-2, 2, 80)
        y = np.linspace(-2, 2, 80)
        z = x + y
        xi = np.linspace(-1, 1, 100)
        yi = np.linspace(-2, 2, 100)
        tck = bisplrep(x, y, z)
        res1 = bisplev(xi, yi, tck)
        interp_ = SmoothBivariateSpline(x, y, z)
        res2 = interp_(xi, yi)
        assert_almost_equal(res1, res2)

    def test_invalid_input(self):
        with assert_raises(ValueError) as info:
            x = np.linspace(1.0, 10.0)
            y = np.linspace(1.0, 10.0)
            z = np.linspace(1.0, 10.0, num=10)
            SmoothBivariateSpline(x, y, z)
        assert 'x, y, and z should have a same length' in str(info.value)
        with assert_raises(ValueError) as info:
            x = np.linspace(1.0, 10.0)
            y = np.linspace(1.0, 10.0)
            z = np.linspace(1.0, 10.0)
            w = np.linspace(1.0, 10.0, num=20)
            SmoothBivariateSpline(x, y, z, w=w)
        assert 'x, y, z, and w should have a same length' in str(info.value)
        with assert_raises(ValueError) as info:
            w = np.linspace(-1.0, 10.0)
            SmoothBivariateSpline(x, y, z, w=w)
        assert 'w should be positive' in str(info.value)
        with assert_raises(ValueError) as info:
            bbox = (-100, 100, -100)
            SmoothBivariateSpline(x, y, z, bbox=bbox)
        assert 'bbox shape should be (4,)' in str(info.value)
        with assert_raises(ValueError) as info:
            SmoothBivariateSpline(x, y, z, kx=10, ky=10)
        assert 'The length of x, y and z should be at least (kx+1) * (ky+1)' in str(info.value)
        with assert_raises(ValueError) as info:
            SmoothBivariateSpline(x, y, z, s=-1.0)
        assert 's should be s >= 0.0' in str(info.value)
        with assert_raises(ValueError) as exc_info:
            SmoothBivariateSpline(x, y, z, eps=0.0)
        assert 'eps should be between (0, 1)' in str(exc_info.value)
        with assert_raises(ValueError) as exc_info:
            SmoothBivariateSpline(x, y, z, eps=1.0)
        assert 'eps should be between (0, 1)' in str(exc_info.value)

    def test_array_like_input(self):
        x = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        y = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        z = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3])
        w = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        bbox = np.array([1.0, 3.0, 1.0, 3.0])
        spl1 = SmoothBivariateSpline(x, y, z, w=w, bbox=bbox, kx=1, ky=1)
        spl2 = SmoothBivariateSpline(x.tolist(), y.tolist(), z.tolist(), bbox=bbox.tolist(), w=w.tolist(), kx=1, ky=1)
        assert_allclose(spl1(0.1, 0.5), spl2(0.1, 0.5))