import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
class TestLSQSphereBivariateSpline:

    def setup_method(self):
        ntheta, nphi = (70, 90)
        theta = linspace(0.5 / (ntheta - 1), 1 - 0.5 / (ntheta - 1), ntheta) * pi
        phi = linspace(0.5 / (nphi - 1), 1 - 0.5 / (nphi - 1), nphi) * 2.0 * pi
        data = ones((theta.shape[0], phi.shape[0]))
        knotst = theta[::5]
        knotsp = phi[::5]
        knotdata = data[::5, ::5]
        lats, lons = meshgrid(theta, phi)
        lut_lsq = LSQSphereBivariateSpline(lats.ravel(), lons.ravel(), data.T.ravel(), knotst, knotsp)
        self.lut_lsq = lut_lsq
        self.data = knotdata
        self.new_lons, self.new_lats = (knotsp, knotst)

    def test_linear_constant(self):
        assert_almost_equal(self.lut_lsq.get_residual(), 0.0)
        assert_array_almost_equal(self.lut_lsq(self.new_lats, self.new_lons), self.data)

    def test_empty_input(self):
        assert_array_almost_equal(self.lut_lsq([], []), np.zeros((0, 0)))
        assert_array_almost_equal(self.lut_lsq([], [], grid=False), np.zeros((0,)))

    def test_invalid_input(self):
        ntheta, nphi = (70, 90)
        theta = linspace(0.5 / (ntheta - 1), 1 - 0.5 / (ntheta - 1), ntheta) * pi
        phi = linspace(0.5 / (nphi - 1), 1 - 0.5 / (nphi - 1), nphi) * 2.0 * pi
        data = ones((theta.shape[0], phi.shape[0]))
        knotst = theta[::5]
        knotsp = phi[::5]
        with assert_raises(ValueError) as exc_info:
            invalid_theta = linspace(-0.1, 1.0, num=ntheta) * pi
            invalid_lats, lons = meshgrid(invalid_theta, phi)
            LSQSphereBivariateSpline(invalid_lats.ravel(), lons.ravel(), data.T.ravel(), knotst, knotsp)
        assert 'theta should be between [0, pi]' in str(exc_info.value)
        with assert_raises(ValueError) as exc_info:
            invalid_theta = linspace(0.1, 1.1, num=ntheta) * pi
            invalid_lats, lons = meshgrid(invalid_theta, phi)
            LSQSphereBivariateSpline(invalid_lats.ravel(), lons.ravel(), data.T.ravel(), knotst, knotsp)
        assert 'theta should be between [0, pi]' in str(exc_info.value)
        with assert_raises(ValueError) as exc_info:
            invalid_phi = linspace(-0.1, 1.0, num=ntheta) * 2.0 * pi
            lats, invalid_lons = meshgrid(theta, invalid_phi)
            LSQSphereBivariateSpline(lats.ravel(), invalid_lons.ravel(), data.T.ravel(), knotst, knotsp)
        assert 'phi should be between [0, 2pi]' in str(exc_info.value)
        with assert_raises(ValueError) as exc_info:
            invalid_phi = linspace(0.0, 1.1, num=ntheta) * 2.0 * pi
            lats, invalid_lons = meshgrid(theta, invalid_phi)
            LSQSphereBivariateSpline(lats.ravel(), invalid_lons.ravel(), data.T.ravel(), knotst, knotsp)
        assert 'phi should be between [0, 2pi]' in str(exc_info.value)
        lats, lons = meshgrid(theta, phi)
        with assert_raises(ValueError) as exc_info:
            invalid_knotst = np.copy(knotst)
            invalid_knotst[0] = -0.1
            LSQSphereBivariateSpline(lats.ravel(), lons.ravel(), data.T.ravel(), invalid_knotst, knotsp)
        assert 'tt should be between (0, pi)' in str(exc_info.value)
        with assert_raises(ValueError) as exc_info:
            invalid_knotst = np.copy(knotst)
            invalid_knotst[0] = pi
            LSQSphereBivariateSpline(lats.ravel(), lons.ravel(), data.T.ravel(), invalid_knotst, knotsp)
        assert 'tt should be between (0, pi)' in str(exc_info.value)
        with assert_raises(ValueError) as exc_info:
            invalid_knotsp = np.copy(knotsp)
            invalid_knotsp[0] = -0.1
            LSQSphereBivariateSpline(lats.ravel(), lons.ravel(), data.T.ravel(), knotst, invalid_knotsp)
        assert 'tp should be between (0, 2pi)' in str(exc_info.value)
        with assert_raises(ValueError) as exc_info:
            invalid_knotsp = np.copy(knotsp)
            invalid_knotsp[0] = 2 * pi
            LSQSphereBivariateSpline(lats.ravel(), lons.ravel(), data.T.ravel(), knotst, invalid_knotsp)
        assert 'tp should be between (0, 2pi)' in str(exc_info.value)
        with assert_raises(ValueError) as exc_info:
            invalid_w = array([-1.0, 1.0, 1.5, 0.5, 1.0, 1.5, 0.5, 1.0, 1.0])
            LSQSphereBivariateSpline(lats.ravel(), lons.ravel(), data.T.ravel(), knotst, knotsp, w=invalid_w)
        assert 'w should be positive' in str(exc_info.value)
        with assert_raises(ValueError) as exc_info:
            LSQSphereBivariateSpline(lats.ravel(), lons.ravel(), data.T.ravel(), knotst, knotsp, eps=0.0)
        assert 'eps should be between (0, 1)' in str(exc_info.value)
        with assert_raises(ValueError) as exc_info:
            LSQSphereBivariateSpline(lats.ravel(), lons.ravel(), data.T.ravel(), knotst, knotsp, eps=1.0)
        assert 'eps should be between (0, 1)' in str(exc_info.value)

    def test_array_like_input(self):
        ntheta, nphi = (70, 90)
        theta = linspace(0.5 / (ntheta - 1), 1 - 0.5 / (ntheta - 1), ntheta) * pi
        phi = linspace(0.5 / (nphi - 1), 1 - 0.5 / (nphi - 1), nphi) * 2.0 * pi
        lats, lons = meshgrid(theta, phi)
        data = ones((theta.shape[0], phi.shape[0]))
        knotst = theta[::5]
        knotsp = phi[::5]
        w = ones(lats.ravel().shape[0])
        spl1 = LSQSphereBivariateSpline(lats.ravel(), lons.ravel(), data.T.ravel(), knotst, knotsp, w=w)
        spl2 = LSQSphereBivariateSpline(lats.ravel().tolist(), lons.ravel().tolist(), data.T.ravel().tolist(), knotst.tolist(), knotsp.tolist(), w=w.tolist())
        assert_array_almost_equal(spl1(1.0, 1.0), spl2(1.0, 1.0))