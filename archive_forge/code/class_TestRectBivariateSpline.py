import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
class TestRectBivariateSpline:

    def test_defaults(self):
        x = array([1, 2, 3, 4, 5])
        y = array([1, 2, 3, 4, 5])
        z = array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1], [1, 2, 2, 2, 1], [1, 2, 1, 2, 1]])
        lut = RectBivariateSpline(x, y, z)
        assert_array_almost_equal(lut(x, y), z)

    def test_evaluate(self):
        x = array([1, 2, 3, 4, 5])
        y = array([1, 2, 3, 4, 5])
        z = array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1], [1, 2, 2, 2, 1], [1, 2, 1, 2, 1]])
        lut = RectBivariateSpline(x, y, z)
        xi = [1, 2.3, 5.3, 0.5, 3.3, 1.2, 3]
        yi = [1, 3.3, 1.2, 4.0, 5.0, 1.0, 3]
        zi = lut.ev(xi, yi)
        zi2 = array([lut(xp, yp)[0, 0] for xp, yp in zip(xi, yi)])
        assert_almost_equal(zi, zi2)

    def test_derivatives_grid(self):
        x = array([1, 2, 3, 4, 5])
        y = array([1, 2, 3, 4, 5])
        z = array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1], [1, 2, 2, 2, 1], [1, 2, 1, 2, 1]])
        dx = array([[0, 0, -20, 0, 0], [0, 0, 13, 0, 0], [0, 0, 4, 0, 0], [0, 0, -11, 0, 0], [0, 0, 4, 0, 0]]) / 6.0
        dy = array([[4, -1, 0, 1, -4], [4, -1, 0, 1, -4], [0, 1.5, 0, -1.5, 0], [2, 0.25, 0, -0.25, -2], [4, -1, 0, 1, -4]])
        dxdy = array([[40, -25, 0, 25, -40], [-26, 16.25, 0, -16.25, 26], [-8, 5, 0, -5, 8], [22, -13.75, 0, 13.75, -22], [-8, 5, 0, -5, 8]]) / 6.0
        lut = RectBivariateSpline(x, y, z)
        assert_array_almost_equal(lut(x, y, dx=1), dx)
        assert_array_almost_equal(lut(x, y, dy=1), dy)
        assert_array_almost_equal(lut(x, y, dx=1, dy=1), dxdy)

    def test_derivatives(self):
        x = array([1, 2, 3, 4, 5])
        y = array([1, 2, 3, 4, 5])
        z = array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1], [1, 2, 2, 2, 1], [1, 2, 1, 2, 1]])
        dx = array([0, 0, 2.0 / 3, 0, 0])
        dy = array([4, -1, 0, -0.25, -4])
        dxdy = array([160, 65, 0, 55, 32]) / 24.0
        lut = RectBivariateSpline(x, y, z)
        assert_array_almost_equal(lut(x, y, dx=1, grid=False), dx)
        assert_array_almost_equal(lut(x, y, dy=1, grid=False), dy)
        assert_array_almost_equal(lut(x, y, dx=1, dy=1, grid=False), dxdy)

    def test_partial_derivative_method_grid(self):
        x = array([1, 2, 3, 4, 5])
        y = array([1, 2, 3, 4, 5])
        z = array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1], [1, 2, 2, 2, 1], [1, 2, 1, 2, 1]])
        dx = array([[0, 0, -20, 0, 0], [0, 0, 13, 0, 0], [0, 0, 4, 0, 0], [0, 0, -11, 0, 0], [0, 0, 4, 0, 0]]) / 6.0
        dy = array([[4, -1, 0, 1, -4], [4, -1, 0, 1, -4], [0, 1.5, 0, -1.5, 0], [2, 0.25, 0, -0.25, -2], [4, -1, 0, 1, -4]])
        dxdy = array([[40, -25, 0, 25, -40], [-26, 16.25, 0, -16.25, 26], [-8, 5, 0, -5, 8], [22, -13.75, 0, 13.75, -22], [-8, 5, 0, -5, 8]]) / 6.0
        lut = RectBivariateSpline(x, y, z)
        assert_array_almost_equal(lut.partial_derivative(1, 0)(x, y), dx)
        assert_array_almost_equal(lut.partial_derivative(0, 1)(x, y), dy)
        assert_array_almost_equal(lut.partial_derivative(1, 1)(x, y), dxdy)

    def test_partial_derivative_method(self):
        x = array([1, 2, 3, 4, 5])
        y = array([1, 2, 3, 4, 5])
        z = array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1], [1, 2, 2, 2, 1], [1, 2, 1, 2, 1]])
        dx = array([0, 0, 2.0 / 3, 0, 0])
        dy = array([4, -1, 0, -0.25, -4])
        dxdy = array([160, 65, 0, 55, 32]) / 24.0
        lut = RectBivariateSpline(x, y, z)
        assert_array_almost_equal(lut.partial_derivative(1, 0)(x, y, grid=False), dx)
        assert_array_almost_equal(lut.partial_derivative(0, 1)(x, y, grid=False), dy)
        assert_array_almost_equal(lut.partial_derivative(1, 1)(x, y, grid=False), dxdy)

    def test_partial_derivative_order_too_large(self):
        x = array([0, 1, 2, 3, 4], dtype=float)
        y = x.copy()
        z = ones((x.size, y.size))
        lut = RectBivariateSpline(x, y, z)
        with assert_raises(ValueError):
            lut.partial_derivative(4, 1)

    def test_broadcast(self):
        x = array([1, 2, 3, 4, 5])
        y = array([1, 2, 3, 4, 5])
        z = array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1], [1, 2, 2, 2, 1], [1, 2, 1, 2, 1]])
        lut = RectBivariateSpline(x, y, z)
        assert_allclose(lut(x, y), lut(x[:, None], y[None, :], grid=False))

    def test_invalid_input(self):
        with assert_raises(ValueError) as info:
            x = array([6, 2, 3, 4, 5])
            y = array([1, 2, 3, 4, 5])
            z = array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1], [1, 2, 2, 2, 1], [1, 2, 1, 2, 1]])
            RectBivariateSpline(x, y, z)
        assert 'x must be strictly increasing' in str(info.value)
        with assert_raises(ValueError) as info:
            x = array([1, 2, 3, 4, 5])
            y = array([2, 2, 3, 4, 5])
            z = array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1], [1, 2, 2, 2, 1], [1, 2, 1, 2, 1]])
            RectBivariateSpline(x, y, z)
        assert 'y must be strictly increasing' in str(info.value)
        with assert_raises(ValueError) as info:
            x = array([1, 2, 3, 4, 5])
            y = array([1, 2, 3, 4, 5])
            z = array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1], [1, 2, 2, 2, 1]])
            RectBivariateSpline(x, y, z)
        assert 'x dimension of z must have same number of elements as x' in str(info.value)
        with assert_raises(ValueError) as info:
            x = array([1, 2, 3, 4, 5])
            y = array([1, 2, 3, 4, 5])
            z = array([[1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 3, 2], [1, 2, 2, 2], [1, 2, 1, 2]])
            RectBivariateSpline(x, y, z)
        assert 'y dimension of z must have same number of elements as y' in str(info.value)
        with assert_raises(ValueError) as info:
            x = array([1, 2, 3, 4, 5])
            y = array([1, 2, 3, 4, 5])
            z = array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1], [1, 2, 2, 2, 1], [1, 2, 1, 2, 1]])
            bbox = (-100, 100, -100)
            RectBivariateSpline(x, y, z, bbox=bbox)
        assert 'bbox shape should be (4,)' in str(info.value)
        with assert_raises(ValueError) as info:
            RectBivariateSpline(x, y, z, s=-1.0)
        assert 's should be s >= 0.0' in str(info.value)

    def test_array_like_input(self):
        x = array([1, 2, 3, 4, 5])
        y = array([1, 2, 3, 4, 5])
        z = array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1], [1, 2, 2, 2, 1], [1, 2, 1, 2, 1]])
        bbox = array([1, 5, 1, 5])
        spl1 = RectBivariateSpline(x, y, z, bbox=bbox)
        spl2 = RectBivariateSpline(x.tolist(), y.tolist(), z.tolist(), bbox=bbox.tolist())
        assert_array_almost_equal(spl1(1.0, 1.0), spl2(1.0, 1.0))

    def test_not_increasing_input(self):
        NSamp = 20
        Theta = np.random.uniform(0, np.pi, NSamp)
        Phi = np.random.uniform(0, 2 * np.pi, NSamp)
        Data = np.ones(NSamp)
        Interpolator = SmoothSphereBivariateSpline(Theta, Phi, Data, s=3.5)
        NLon = 6
        NLat = 3
        GridPosLats = np.arange(NLat) / NLat * np.pi
        GridPosLons = np.arange(NLon) / NLon * 2 * np.pi
        Interpolator(GridPosLats, GridPosLons)
        nonGridPosLats = GridPosLats.copy()
        nonGridPosLats[2] = 0.001
        with assert_raises(ValueError) as exc_info:
            Interpolator(nonGridPosLats, GridPosLons)
        assert 'x must be strictly increasing' in str(exc_info.value)
        nonGridPosLons = GridPosLons.copy()
        nonGridPosLons[2] = 0.001
        with assert_raises(ValueError) as exc_info:
            Interpolator(GridPosLats, nonGridPosLons)
        assert 'y must be strictly increasing' in str(exc_info.value)