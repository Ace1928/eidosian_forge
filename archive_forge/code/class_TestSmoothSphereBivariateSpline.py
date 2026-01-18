import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
class TestSmoothSphereBivariateSpline:

    def setup_method(self):
        theta = array([0.25 * pi, 0.25 * pi, 0.25 * pi, 0.5 * pi, 0.5 * pi, 0.5 * pi, 0.75 * pi, 0.75 * pi, 0.75 * pi])
        phi = array([0.5 * pi, pi, 1.5 * pi, 0.5 * pi, pi, 1.5 * pi, 0.5 * pi, pi, 1.5 * pi])
        r = array([3, 3, 3, 3, 3, 3, 3, 3, 3])
        self.lut = SmoothSphereBivariateSpline(theta, phi, r, s=10000000000.0)

    def test_linear_constant(self):
        assert_almost_equal(self.lut.get_residual(), 0.0)
        assert_array_almost_equal(self.lut([1, 1.5, 2], [1, 1.5]), [[3, 3], [3, 3], [3, 3]])

    def test_empty_input(self):
        assert_array_almost_equal(self.lut([], []), np.zeros((0, 0)))
        assert_array_almost_equal(self.lut([], [], grid=False), np.zeros((0,)))

    def test_invalid_input(self):
        theta = array([0.25 * pi, 0.25 * pi, 0.25 * pi, 0.5 * pi, 0.5 * pi, 0.5 * pi, 0.75 * pi, 0.75 * pi, 0.75 * pi])
        phi = array([0.5 * pi, pi, 1.5 * pi, 0.5 * pi, pi, 1.5 * pi, 0.5 * pi, pi, 1.5 * pi])
        r = array([3, 3, 3, 3, 3, 3, 3, 3, 3])
        with assert_raises(ValueError) as exc_info:
            invalid_theta = array([-0.1 * pi, 0.25 * pi, 0.25 * pi, 0.5 * pi, 0.5 * pi, 0.5 * pi, 0.75 * pi, 0.75 * pi, 0.75 * pi])
            SmoothSphereBivariateSpline(invalid_theta, phi, r, s=10000000000.0)
        assert 'theta should be between [0, pi]' in str(exc_info.value)
        with assert_raises(ValueError) as exc_info:
            invalid_theta = array([0.25 * pi, 0.25 * pi, 0.25 * pi, 0.5 * pi, 0.5 * pi, 0.5 * pi, 0.75 * pi, 0.75 * pi, 1.1 * pi])
            SmoothSphereBivariateSpline(invalid_theta, phi, r, s=10000000000.0)
        assert 'theta should be between [0, pi]' in str(exc_info.value)
        with assert_raises(ValueError) as exc_info:
            invalid_phi = array([-0.1 * pi, pi, 1.5 * pi, 0.5 * pi, pi, 1.5 * pi, 0.5 * pi, pi, 1.5 * pi])
            SmoothSphereBivariateSpline(theta, invalid_phi, r, s=10000000000.0)
        assert 'phi should be between [0, 2pi]' in str(exc_info.value)
        with assert_raises(ValueError) as exc_info:
            invalid_phi = array([1.0 * pi, pi, 1.5 * pi, 0.5 * pi, pi, 1.5 * pi, 0.5 * pi, pi, 2.1 * pi])
            SmoothSphereBivariateSpline(theta, invalid_phi, r, s=10000000000.0)
        assert 'phi should be between [0, 2pi]' in str(exc_info.value)
        with assert_raises(ValueError) as exc_info:
            invalid_w = array([-1.0, 1.0, 1.5, 0.5, 1.0, 1.5, 0.5, 1.0, 1.0])
            SmoothSphereBivariateSpline(theta, phi, r, w=invalid_w, s=10000000000.0)
        assert 'w should be positive' in str(exc_info.value)
        with assert_raises(ValueError) as exc_info:
            SmoothSphereBivariateSpline(theta, phi, r, s=-1.0)
        assert 's should be positive' in str(exc_info.value)
        with assert_raises(ValueError) as exc_info:
            SmoothSphereBivariateSpline(theta, phi, r, eps=-1.0)
        assert 'eps should be between (0, 1)' in str(exc_info.value)
        with assert_raises(ValueError) as exc_info:
            SmoothSphereBivariateSpline(theta, phi, r, eps=1.0)
        assert 'eps should be between (0, 1)' in str(exc_info.value)

    def test_array_like_input(self):
        theta = np.array([0.25 * pi, 0.25 * pi, 0.25 * pi, 0.5 * pi, 0.5 * pi, 0.5 * pi, 0.75 * pi, 0.75 * pi, 0.75 * pi])
        phi = np.array([0.5 * pi, pi, 1.5 * pi, 0.5 * pi, pi, 1.5 * pi, 0.5 * pi, pi, 1.5 * pi])
        r = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3])
        w = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        spl1 = SmoothSphereBivariateSpline(theta, phi, r, w=w, s=10000000000.0)
        spl2 = SmoothSphereBivariateSpline(theta.tolist(), phi.tolist(), r.tolist(), w=w.tolist(), s=10000000000.0)
        assert_array_almost_equal(spl1(1.0, 1.0), spl2(1.0, 1.0))