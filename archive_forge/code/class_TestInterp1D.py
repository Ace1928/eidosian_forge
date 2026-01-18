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
class TestInterp1D:

    def setup_method(self):
        self.x5 = np.arange(5.0)
        self.x10 = np.arange(10.0)
        self.y10 = np.arange(10.0)
        self.x25 = self.x10.reshape((2, 5))
        self.x2 = np.arange(2.0)
        self.y2 = np.arange(2.0)
        self.x1 = np.array([0.0])
        self.y1 = np.array([0.0])
        self.y210 = np.arange(20.0).reshape((2, 10))
        self.y102 = np.arange(20.0).reshape((10, 2))
        self.y225 = np.arange(20.0).reshape((2, 2, 5))
        self.y25 = np.arange(10.0).reshape((2, 5))
        self.y235 = np.arange(30.0).reshape((2, 3, 5))
        self.y325 = np.arange(30.0).reshape((3, 2, 5))
        self.y210_edge_updated = np.arange(20.0).reshape((2, 10))
        self.y210_edge_updated[:, 0] = 30
        self.y210_edge_updated[:, -1] = -30
        self.y102_edge_updated = np.arange(20.0).reshape((10, 2))
        self.y102_edge_updated[0, :] = 30
        self.y102_edge_updated[-1, :] = -30
        self.fill_value = -100.0

    def test_validation(self):
        for kind in ('nearest', 'nearest-up', 'zero', 'linear', 'slinear', 'quadratic', 'cubic', 'previous', 'next'):
            interp1d(self.x10, self.y10, kind=kind)
            interp1d(self.x10, self.y10, kind=kind, fill_value='extrapolate')
        interp1d(self.x10, self.y10, kind='linear', fill_value=(-1, 1))
        interp1d(self.x10, self.y10, kind='linear', fill_value=np.array([-1]))
        interp1d(self.x10, self.y10, kind='linear', fill_value=(-1,))
        interp1d(self.x10, self.y10, kind='linear', fill_value=-1)
        interp1d(self.x10, self.y10, kind='linear', fill_value=(-1, -1))
        interp1d(self.x10, self.y10, kind=0)
        interp1d(self.x10, self.y10, kind=1)
        interp1d(self.x10, self.y10, kind=2)
        interp1d(self.x10, self.y10, kind=3)
        interp1d(self.x10, self.y210, kind='linear', axis=-1, fill_value=(-1, -1))
        interp1d(self.x2, self.y210, kind='linear', axis=0, fill_value=np.ones(10))
        interp1d(self.x2, self.y210, kind='linear', axis=0, fill_value=(np.ones(10), np.ones(10)))
        interp1d(self.x2, self.y210, kind='linear', axis=0, fill_value=(np.ones(10), -1))
        assert_raises(ValueError, interp1d, self.x25, self.y10)
        assert_raises(ValueError, interp1d, self.x10, np.array(0))
        assert_raises(ValueError, interp1d, self.x10, self.y2)
        assert_raises(ValueError, interp1d, self.x2, self.y10)
        assert_raises(ValueError, interp1d, self.x10, self.y102)
        interp1d(self.x10, self.y210)
        interp1d(self.x10, self.y102, axis=0)
        assert_raises(ValueError, interp1d, self.x1, self.y10)
        assert_raises(ValueError, interp1d, self.x10, self.y1)
        assert_raises(ValueError, interp1d, self.x10, self.y10, kind='linear', fill_value=(-1, -1, -1))
        assert_raises(ValueError, interp1d, self.x10, self.y10, kind='linear', fill_value=[-1, -1, -1])
        assert_raises(ValueError, interp1d, self.x10, self.y10, kind='linear', fill_value=np.array((-1, -1, -1)))
        assert_raises(ValueError, interp1d, self.x10, self.y10, kind='linear', fill_value=[[-1]])
        assert_raises(ValueError, interp1d, self.x10, self.y10, kind='linear', fill_value=[-1, -1])
        assert_raises(ValueError, interp1d, self.x10, self.y10, kind='linear', fill_value=np.array([]))
        assert_raises(ValueError, interp1d, self.x10, self.y10, kind='linear', fill_value=())
        assert_raises(ValueError, interp1d, self.x2, self.y210, kind='linear', axis=0, fill_value=[-1, -1])
        assert_raises(ValueError, interp1d, self.x2, self.y210, kind='linear', axis=0, fill_value=(0.0, [-1, -1]))

    def test_init(self):
        assert_(interp1d(self.x10, self.y10).copy)
        assert_(not interp1d(self.x10, self.y10, copy=False).copy)
        assert_(interp1d(self.x10, self.y10).bounds_error)
        assert_(not interp1d(self.x10, self.y10, bounds_error=False).bounds_error)
        assert_(np.isnan(interp1d(self.x10, self.y10).fill_value))
        assert_equal(interp1d(self.x10, self.y10, fill_value=3.0).fill_value, 3.0)
        assert_equal(interp1d(self.x10, self.y10, fill_value=(1.0, 2.0)).fill_value, (1.0, 2.0))
        assert_equal(interp1d(self.x10, self.y10).axis, 0)
        assert_equal(interp1d(self.x10, self.y210).axis, 1)
        assert_equal(interp1d(self.x10, self.y102, axis=0).axis, 0)
        assert_array_equal(interp1d(self.x10, self.y10).x, self.x10)
        assert_array_equal(interp1d(self.x10, self.y10).y, self.y10)
        assert_array_equal(interp1d(self.x10, self.y210).y, self.y210)

    def test_assume_sorted(self):
        interp10 = interp1d(self.x10, self.y10)
        interp10_unsorted = interp1d(self.x10[::-1], self.y10[::-1])
        assert_array_almost_equal(interp10_unsorted(self.x10), self.y10)
        assert_array_almost_equal(interp10_unsorted(1.2), np.array([1.2]))
        assert_array_almost_equal(interp10_unsorted([2.4, 5.6, 6.0]), interp10([2.4, 5.6, 6.0]))
        interp10_assume_kw = interp1d(self.x10[::-1], self.y10[::-1], assume_sorted=False)
        assert_array_almost_equal(interp10_assume_kw(self.x10), self.y10)
        interp10_assume_kw2 = interp1d(self.x10[::-1], self.y10[::-1], assume_sorted=True)
        assert_raises(ValueError, interp10_assume_kw2, self.x10)
        interp10_y_2d = interp1d(self.x10, self.y210)
        interp10_y_2d_unsorted = interp1d(self.x10[::-1], self.y210[:, ::-1])
        assert_array_almost_equal(interp10_y_2d(self.x10), interp10_y_2d_unsorted(self.x10))

    def test_linear(self):
        for kind in ['linear', 'slinear']:
            self._check_linear(kind)

    def _check_linear(self, kind):
        interp10 = interp1d(self.x10, self.y10, kind=kind)
        assert_array_almost_equal(interp10(self.x10), self.y10)
        assert_array_almost_equal(interp10(1.2), np.array([1.2]))
        assert_array_almost_equal(interp10([2.4, 5.6, 6.0]), np.array([2.4, 5.6, 6.0]))
        extrapolator = interp1d(self.x10, self.y10, kind=kind, fill_value='extrapolate')
        assert_allclose(extrapolator([-1.0, 0, 9, 11]), [-1, 0, 9, 11], rtol=1e-14)
        opts = dict(kind=kind, fill_value='extrapolate', bounds_error=True)
        assert_raises(ValueError, interp1d, self.x10, self.y10, **opts)

    def test_linear_dtypes(self):
        for dtyp in [np.float16, np.float32, np.float64, np.longdouble]:
            x = np.arange(8, dtype=dtyp)
            y = x
            yp = interp1d(x, y, kind='linear')(x)
            assert_equal(yp.dtype, dtyp)
            assert_allclose(yp, y, atol=1e-15)
        x = [0, 1, 2]
        y = [np.nan, 0, 1]
        yp = interp1d(x, y)(x)
        assert_allclose(yp, y, atol=1e-15)

    def test_slinear_dtypes(self):
        dt_r = [np.float16, np.float32, np.float64]
        dt_rc = dt_r + [np.complex64, np.complex128]
        spline_kinds = ['slinear', 'zero', 'quadratic', 'cubic']
        for dtx in dt_r:
            x = np.arange(0, 10, dtype=dtx)
            for dty in dt_rc:
                y = np.exp(-x / 3.0).astype(dty)
                for dtn in dt_r:
                    xnew = x.astype(dtn)
                    for kind in spline_kinds:
                        f = interp1d(x, y, kind=kind, bounds_error=False)
                        assert_allclose(f(xnew), y, atol=1e-07, err_msg=f'{dtx}, {dty} {dtn}')

    def test_cubic(self):
        interp10 = interp1d(self.x10, self.y10, kind='cubic')
        assert_array_almost_equal(interp10(self.x10), self.y10)
        assert_array_almost_equal(interp10(1.2), np.array([1.2]))
        assert_array_almost_equal(interp10(1.5), np.array([1.5]))
        assert_array_almost_equal(interp10([2.4, 5.6, 6.0]), np.array([2.4, 5.6, 6.0]))

    def test_nearest(self):
        interp10 = interp1d(self.x10, self.y10, kind='nearest')
        assert_array_almost_equal(interp10(self.x10), self.y10)
        assert_array_almost_equal(interp10(1.2), np.array(1.0))
        assert_array_almost_equal(interp10(1.5), np.array(1.0))
        assert_array_almost_equal(interp10([2.4, 5.6, 6.0]), np.array([2.0, 6.0, 6.0]))
        extrapolator = interp1d(self.x10, self.y10, kind='nearest', fill_value='extrapolate')
        assert_allclose(extrapolator([-1.0, 0, 9, 11]), [0, 0, 9, 9], rtol=1e-14)
        opts = dict(kind='nearest', fill_value='extrapolate', bounds_error=True)
        assert_raises(ValueError, interp1d, self.x10, self.y10, **opts)

    def test_nearest_up(self):
        interp10 = interp1d(self.x10, self.y10, kind='nearest-up')
        assert_array_almost_equal(interp10(self.x10), self.y10)
        assert_array_almost_equal(interp10(1.2), np.array(1.0))
        assert_array_almost_equal(interp10(1.5), np.array(2.0))
        assert_array_almost_equal(interp10([2.4, 5.6, 6.0]), np.array([2.0, 6.0, 6.0]))
        extrapolator = interp1d(self.x10, self.y10, kind='nearest-up', fill_value='extrapolate')
        assert_allclose(extrapolator([-1.0, 0, 9, 11]), [0, 0, 9, 9], rtol=1e-14)
        opts = dict(kind='nearest-up', fill_value='extrapolate', bounds_error=True)
        assert_raises(ValueError, interp1d, self.x10, self.y10, **opts)

    def test_previous(self):
        interp10 = interp1d(self.x10, self.y10, kind='previous')
        assert_array_almost_equal(interp10(self.x10), self.y10)
        assert_array_almost_equal(interp10(1.2), np.array(1.0))
        assert_array_almost_equal(interp10(1.5), np.array(1.0))
        assert_array_almost_equal(interp10([2.4, 5.6, 6.0]), np.array([2.0, 5.0, 6.0]))
        extrapolator = interp1d(self.x10, self.y10, kind='previous', fill_value='extrapolate')
        assert_allclose(extrapolator([-1.0, 0, 9, 11]), [np.nan, 0, 9, 9], rtol=1e-14)
        interpolator1D = interp1d(self.x10, self.y10, kind='previous', fill_value='extrapolate')
        assert_allclose(interpolator1D([-1, -2, 5, 8, 12, 25]), [np.nan, np.nan, 5, 8, 9, 9])
        interpolator2D = interp1d(self.x10, self.y210, kind='previous', fill_value='extrapolate')
        assert_allclose(interpolator2D([-1, -2, 5, 8, 12, 25]), [[np.nan, np.nan, 5, 8, 9, 9], [np.nan, np.nan, 15, 18, 19, 19]])
        interpolator2DAxis0 = interp1d(self.x10, self.y102, kind='previous', axis=0, fill_value='extrapolate')
        assert_allclose(interpolator2DAxis0([-2, 5, 12]), [[np.nan, np.nan], [10, 11], [18, 19]])
        opts = dict(kind='previous', fill_value='extrapolate', bounds_error=True)
        assert_raises(ValueError, interp1d, self.x10, self.y10, **opts)
        interpolator1D = interp1d([0, 1, 2], [0, 1, -1], kind='previous', fill_value='extrapolate', assume_sorted=True)
        assert_allclose(interpolator1D([-2, -1, 0, 1, 2, 3, 5]), [np.nan, np.nan, 0, 1, -1, -1, -1])
        interpolator1D = interp1d([2, 0, 1], [-1, 0, 1], kind='previous', fill_value='extrapolate', assume_sorted=False)
        assert_allclose(interpolator1D([-2, -1, 0, 1, 2, 3, 5]), [np.nan, np.nan, 0, 1, -1, -1, -1])
        interpolator2D = interp1d(self.x10, self.y210_edge_updated, kind='previous', fill_value='extrapolate')
        assert_allclose(interpolator2D([-1, -2, 5, 8, 12, 25]), [[np.nan, np.nan, 5, 8, -30, -30], [np.nan, np.nan, 15, 18, -30, -30]])
        interpolator2DAxis0 = interp1d(self.x10, self.y102_edge_updated, kind='previous', axis=0, fill_value='extrapolate')
        assert_allclose(interpolator2DAxis0([-2, 5, 12]), [[np.nan, np.nan], [10, 11], [-30, -30]])

    def test_next(self):
        interp10 = interp1d(self.x10, self.y10, kind='next')
        assert_array_almost_equal(interp10(self.x10), self.y10)
        assert_array_almost_equal(interp10(1.2), np.array(2.0))
        assert_array_almost_equal(interp10(1.5), np.array(2.0))
        assert_array_almost_equal(interp10([2.4, 5.6, 6.0]), np.array([3.0, 6.0, 6.0]))
        extrapolator = interp1d(self.x10, self.y10, kind='next', fill_value='extrapolate')
        assert_allclose(extrapolator([-1.0, 0, 9, 11]), [0, 0, 9, np.nan], rtol=1e-14)
        interpolator1D = interp1d(self.x10, self.y10, kind='next', fill_value='extrapolate')
        assert_allclose(interpolator1D([-1, -2, 5, 8, 12, 25]), [0, 0, 5, 8, np.nan, np.nan])
        interpolator2D = interp1d(self.x10, self.y210, kind='next', fill_value='extrapolate')
        assert_allclose(interpolator2D([-1, -2, 5, 8, 12, 25]), [[0, 0, 5, 8, np.nan, np.nan], [10, 10, 15, 18, np.nan, np.nan]])
        interpolator2DAxis0 = interp1d(self.x10, self.y102, kind='next', axis=0, fill_value='extrapolate')
        assert_allclose(interpolator2DAxis0([-2, 5, 12]), [[0, 1], [10, 11], [np.nan, np.nan]])
        opts = dict(kind='next', fill_value='extrapolate', bounds_error=True)
        assert_raises(ValueError, interp1d, self.x10, self.y10, **opts)
        interpolator1D = interp1d([0, 1, 2], [0, 1, -1], kind='next', fill_value='extrapolate', assume_sorted=True)
        assert_allclose(interpolator1D([-2, -1, 0, 1, 2, 3, 5]), [0, 0, 0, 1, -1, np.nan, np.nan])
        interpolator1D = interp1d([2, 0, 1], [-1, 0, 1], kind='next', fill_value='extrapolate', assume_sorted=False)
        assert_allclose(interpolator1D([-2, -1, 0, 1, 2, 3, 5]), [0, 0, 0, 1, -1, np.nan, np.nan])
        interpolator2D = interp1d(self.x10, self.y210_edge_updated, kind='next', fill_value='extrapolate')
        assert_allclose(interpolator2D([-1, -2, 5, 8, 12, 25]), [[30, 30, 5, 8, np.nan, np.nan], [30, 30, 15, 18, np.nan, np.nan]])
        interpolator2DAxis0 = interp1d(self.x10, self.y102_edge_updated, kind='next', axis=0, fill_value='extrapolate')
        assert_allclose(interpolator2DAxis0([-2, 5, 12]), [[30, 30], [10, 11], [np.nan, np.nan]])

    def test_zero(self):
        interp10 = interp1d(self.x10, self.y10, kind='zero')
        assert_array_almost_equal(interp10(self.x10), self.y10)
        assert_array_almost_equal(interp10(1.2), np.array(1.0))
        assert_array_almost_equal(interp10(1.5), np.array(1.0))
        assert_array_almost_equal(interp10([2.4, 5.6, 6.0]), np.array([2.0, 5.0, 6.0]))

    def bounds_check_helper(self, interpolant, test_array, fail_value):
        assert_raises(ValueError, interpolant, test_array)
        try:
            interpolant(test_array)
        except ValueError as err:
            assert f'{fail_value}' in str(err)

    def _bounds_check(self, kind='linear'):
        extrap10 = interp1d(self.x10, self.y10, fill_value=self.fill_value, bounds_error=False, kind=kind)
        assert_array_equal(extrap10(11.2), np.array(self.fill_value))
        assert_array_equal(extrap10(-3.4), np.array(self.fill_value))
        assert_array_equal(extrap10([[[11.2], [-3.4], [12.6], [19.3]]]), np.array(self.fill_value))
        assert_array_equal(extrap10._check_bounds(np.array([-1.0, 0.0, 5.0, 9.0, 11.0])), np.array([[True, False, False, False, False], [False, False, False, False, True]]))
        raises_bounds_error = interp1d(self.x10, self.y10, bounds_error=True, kind=kind)
        self.bounds_check_helper(raises_bounds_error, -1.0, -1.0)
        self.bounds_check_helper(raises_bounds_error, 11.0, 11.0)
        self.bounds_check_helper(raises_bounds_error, [0.0, -1.0, 0.0], -1.0)
        self.bounds_check_helper(raises_bounds_error, [0.0, 1.0, 21.0], 21.0)
        raises_bounds_error([0.0, 5.0, 9.0])

    def _bounds_check_int_nan_fill(self, kind='linear'):
        x = np.arange(10).astype(int)
        y = np.arange(10).astype(int)
        c = interp1d(x, y, kind=kind, fill_value=np.nan, bounds_error=False)
        yi = c(x - 1)
        assert_(np.isnan(yi[0]))
        assert_array_almost_equal(yi, np.r_[np.nan, y[:-1]])

    def test_bounds(self):
        for kind in ('linear', 'cubic', 'nearest', 'previous', 'next', 'slinear', 'zero', 'quadratic'):
            self._bounds_check(kind)
            self._bounds_check_int_nan_fill(kind)

    def _check_fill_value(self, kind):
        interp = interp1d(self.x10, self.y10, kind=kind, fill_value=(-100, 100), bounds_error=False)
        assert_array_almost_equal(interp(10), 100)
        assert_array_almost_equal(interp(-10), -100)
        assert_array_almost_equal(interp([-10, 10]), [-100, 100])
        for y in (self.y235, self.y325, self.y225, self.y25):
            interp = interp1d(self.x5, y, kind=kind, axis=-1, fill_value=100, bounds_error=False)
            assert_array_almost_equal(interp(10), 100)
            assert_array_almost_equal(interp(-10), 100)
            assert_array_almost_equal(interp([-10, 10]), 100)
            interp = interp1d(self.x5, y, kind=kind, axis=-1, fill_value=(-100, 100), bounds_error=False)
            assert_array_almost_equal(interp(10), 100)
            assert_array_almost_equal(interp(-10), -100)
            if y.ndim == 3:
                result = [[[-100, 100]] * y.shape[1]] * y.shape[0]
            else:
                result = [[-100, 100]] * y.shape[0]
            assert_array_almost_equal(interp([-10, 10]), result)
        fill_value = [100, 200, 300]
        for y in (self.y325, self.y225):
            assert_raises(ValueError, interp1d, self.x5, y, kind=kind, axis=-1, fill_value=fill_value, bounds_error=False)
        interp = interp1d(self.x5, self.y235, kind=kind, axis=-1, fill_value=fill_value, bounds_error=False)
        assert_array_almost_equal(interp(10), [[100, 200, 300]] * 2)
        assert_array_almost_equal(interp(-10), [[100, 200, 300]] * 2)
        assert_array_almost_equal(interp([-10, 10]), [[[100, 100], [200, 200], [300, 300]]] * 2)
        fill_value = [100, 200]
        assert_raises(ValueError, interp1d, self.x5, self.y235, kind=kind, axis=-1, fill_value=fill_value, bounds_error=False)
        for y in (self.y225, self.y325, self.y25):
            interp = interp1d(self.x5, y, kind=kind, axis=-1, fill_value=fill_value, bounds_error=False)
            result = [100, 200]
            if y.ndim == 3:
                result = [result] * y.shape[0]
            assert_array_almost_equal(interp(10), result)
            assert_array_almost_equal(interp(-10), result)
            result = [[100, 100], [200, 200]]
            if y.ndim == 3:
                result = [result] * y.shape[0]
            assert_array_almost_equal(interp([-10, 10]), result)
        fill_value = (np.array([-100, -200, -300]), 100)
        for y in (self.y325, self.y225):
            assert_raises(ValueError, interp1d, self.x5, y, kind=kind, axis=-1, fill_value=fill_value, bounds_error=False)
        interp = interp1d(self.x5, self.y235, kind=kind, axis=-1, fill_value=fill_value, bounds_error=False)
        assert_array_almost_equal(interp(10), 100)
        assert_array_almost_equal(interp(-10), [[-100, -200, -300]] * 2)
        assert_array_almost_equal(interp([-10, 10]), [[[-100, 100], [-200, 100], [-300, 100]]] * 2)
        fill_value = (np.array([-100, -200]), 100)
        assert_raises(ValueError, interp1d, self.x5, self.y235, kind=kind, axis=-1, fill_value=fill_value, bounds_error=False)
        for y in (self.y225, self.y325, self.y25):
            interp = interp1d(self.x5, y, kind=kind, axis=-1, fill_value=fill_value, bounds_error=False)
            assert_array_almost_equal(interp(10), 100)
            result = [-100, -200]
            if y.ndim == 3:
                result = [result] * y.shape[0]
            assert_array_almost_equal(interp(-10), result)
            result = [[-100, 100], [-200, 100]]
            if y.ndim == 3:
                result = [result] * y.shape[0]
            assert_array_almost_equal(interp([-10, 10]), result)
        fill_value = ([-100, -200, -300], [100, 200, 300])
        for y in (self.y325, self.y225):
            assert_raises(ValueError, interp1d, self.x5, y, kind=kind, axis=-1, fill_value=fill_value, bounds_error=False)
        for ii in range(2):
            if ii == 1:
                fill_value = tuple((np.array(f) for f in fill_value))
            interp = interp1d(self.x5, self.y235, kind=kind, axis=-1, fill_value=fill_value, bounds_error=False)
            assert_array_almost_equal(interp(10), [[100, 200, 300]] * 2)
            assert_array_almost_equal(interp(-10), [[-100, -200, -300]] * 2)
            assert_array_almost_equal(interp([-10, 10]), [[[-100, 100], [-200, 200], [-300, 300]]] * 2)
        fill_value = ([-100, -200], [100, 200])
        assert_raises(ValueError, interp1d, self.x5, self.y235, kind=kind, axis=-1, fill_value=fill_value, bounds_error=False)
        for y in (self.y325, self.y225, self.y25):
            interp = interp1d(self.x5, y, kind=kind, axis=-1, fill_value=fill_value, bounds_error=False)
            result = [100, 200]
            if y.ndim == 3:
                result = [result] * y.shape[0]
            assert_array_almost_equal(interp(10), result)
            result = [-100, -200]
            if y.ndim == 3:
                result = [result] * y.shape[0]
            assert_array_almost_equal(interp(-10), result)
            result = [[-100, 100], [-200, 200]]
            if y.ndim == 3:
                result = [result] * y.shape[0]
            assert_array_almost_equal(interp([-10, 10]), result)
        fill_value = [[100, 200], [1000, 2000]]
        for y in (self.y235, self.y325, self.y25):
            assert_raises(ValueError, interp1d, self.x5, y, kind=kind, axis=-1, fill_value=fill_value, bounds_error=False)
        for ii in range(2):
            if ii == 1:
                fill_value = np.array(fill_value)
            interp = interp1d(self.x5, self.y225, kind=kind, axis=-1, fill_value=fill_value, bounds_error=False)
            assert_array_almost_equal(interp(10), [[100, 200], [1000, 2000]])
            assert_array_almost_equal(interp(-10), [[100, 200], [1000, 2000]])
            assert_array_almost_equal(interp([-10, 10]), [[[100, 100], [200, 200]], [[1000, 1000], [2000, 2000]]])
        fill_value = ([[-100, -200], [-1000, -2000]], [[100, 200], [1000, 2000]])
        for y in (self.y235, self.y325, self.y25):
            assert_raises(ValueError, interp1d, self.x5, y, kind=kind, axis=-1, fill_value=fill_value, bounds_error=False)
        for ii in range(2):
            if ii == 1:
                fill_value = (np.array(fill_value[0]), np.array(fill_value[1]))
            interp = interp1d(self.x5, self.y225, kind=kind, axis=-1, fill_value=fill_value, bounds_error=False)
            assert_array_almost_equal(interp(10), [[100, 200], [1000, 2000]])
            assert_array_almost_equal(interp(-10), [[-100, -200], [-1000, -2000]])
            assert_array_almost_equal(interp([-10, 10]), [[[-100, 100], [-200, 200]], [[-1000, 1000], [-2000, 2000]]])

    def test_fill_value(self):
        for kind in ('linear', 'nearest', 'cubic', 'slinear', 'quadratic', 'zero', 'previous', 'next'):
            self._check_fill_value(kind)

    def test_fill_value_writeable(self):
        interp = interp1d(self.x10, self.y10, fill_value=123.0)
        assert_equal(interp.fill_value, 123.0)
        interp.fill_value = 321.0
        assert_equal(interp.fill_value, 321.0)

    def _nd_check_interp(self, kind='linear'):
        interp10 = interp1d(self.x10, self.y10, kind=kind)
        assert_array_almost_equal(interp10(np.array([[3.0, 5.0], [2.0, 7.0]])), np.array([[3.0, 5.0], [2.0, 7.0]]))
        assert_(isinstance(interp10(1.2), np.ndarray))
        assert_equal(interp10(1.2).shape, ())
        interp210 = interp1d(self.x10, self.y210, kind=kind)
        assert_array_almost_equal(interp210(1.0), np.array([1.0, 11.0]))
        assert_array_almost_equal(interp210(np.array([1.0, 2.0])), np.array([[1.0, 2.0], [11.0, 12.0]]))
        interp102 = interp1d(self.x10, self.y102, axis=0, kind=kind)
        assert_array_almost_equal(interp102(1.0), np.array([2.0, 3.0]))
        assert_array_almost_equal(interp102(np.array([1.0, 3.0])), np.array([[2.0, 3.0], [6.0, 7.0]]))
        x_new = np.array([[3.0, 5.0], [2.0, 7.0]])
        assert_array_almost_equal(interp210(x_new), np.array([[[3.0, 5.0], [2.0, 7.0]], [[13.0, 15.0], [12.0, 17.0]]]))
        assert_array_almost_equal(interp102(x_new), np.array([[[6.0, 7.0], [10.0, 11.0]], [[4.0, 5.0], [14.0, 15.0]]]))

    def _nd_check_shape(self, kind='linear'):
        a = [4, 5, 6, 7]
        y = np.arange(np.prod(a)).reshape(*a)
        for n, s in enumerate(a):
            x = np.arange(s)
            z = interp1d(x, y, axis=n, kind=kind)
            assert_array_almost_equal(z(x), y, err_msg=kind)
            x2 = np.arange(2 * 3 * 1).reshape((2, 3, 1)) / 12.0
            b = list(a)
            b[n:n + 1] = [2, 3, 1]
            assert_array_almost_equal(z(x2).shape, b, err_msg=kind)

    def test_nd(self):
        for kind in ('linear', 'cubic', 'slinear', 'quadratic', 'nearest', 'zero', 'previous', 'next'):
            self._nd_check_interp(kind)
            self._nd_check_shape(kind)

    def _check_complex(self, dtype=np.complex128, kind='linear'):
        x = np.array([1, 2.5, 3, 3.1, 4, 6.4, 7.9, 8.0, 9.5, 10])
        y = x * x ** (1 + 2j)
        y = y.astype(dtype)
        c = interp1d(x, y, kind=kind)
        assert_array_almost_equal(y[:-1], c(x)[:-1])
        xi = np.linspace(1, 10, 31)
        cr = interp1d(x, y.real, kind=kind)
        ci = interp1d(x, y.imag, kind=kind)
        assert_array_almost_equal(c(xi).real, cr(xi))
        assert_array_almost_equal(c(xi).imag, ci(xi))

    def test_complex(self):
        for kind in ('linear', 'nearest', 'cubic', 'slinear', 'quadratic', 'zero', 'previous', 'next'):
            self._check_complex(np.complex64, kind)
            self._check_complex(np.complex128, kind)

    @pytest.mark.skipif(IS_PYPY, reason='Test not meaningful on PyPy')
    def test_circular_refs(self):
        x = np.linspace(0, 1)
        y = np.linspace(0, 1)
        with assert_deallocated(interp1d, x, y) as interp:
            interp([0.1, 0.2])
            del interp

    def test_overflow_nearest(self):
        for kind in ('nearest', 'previous', 'next'):
            x = np.array([0, 50, 127], dtype=np.int8)
            ii = interp1d(x, x, kind=kind)
            assert_array_almost_equal(ii(x), x)

    def test_local_nans(self):
        x = np.arange(10).astype(float)
        y = x.copy()
        y[6] = np.nan
        for kind in ('zero', 'slinear'):
            ir = interp1d(x, y, kind=kind)
            vals = ir([4.9, 7.0])
            assert_(np.isfinite(vals).all())

    def test_spline_nans(self):
        x = np.arange(8).astype(float)
        y = x.copy()
        yn = y.copy()
        yn[3] = np.nan
        for kind in ['quadratic', 'cubic']:
            ir = interp1d(x, y, kind=kind)
            irn = interp1d(x, yn, kind=kind)
            for xnew in (6, [1, 6], [[1, 6], [3, 5]]):
                xnew = np.asarray(xnew)
                out, outn = (ir(x), irn(x))
                assert_(np.isnan(outn).all())
                assert_equal(out.shape, outn.shape)

    def test_all_nans(self):
        x = np.ones(10) * np.nan
        y = np.arange(10)
        with assert_raises(ValueError):
            interp1d(x, y, kind='cubic')

    def test_read_only(self):
        x = np.arange(0, 10)
        y = np.exp(-x / 3.0)
        xnew = np.arange(0, 9, 0.1)
        for xnew_writeable in (True, False):
            xnew.flags.writeable = xnew_writeable
            x.flags.writeable = False
            for kind in ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'):
                f = interp1d(x, y, kind=kind)
                vals = f(xnew)
                assert_(np.isfinite(vals).all())

    @pytest.mark.parametrize('kind', ('linear', 'nearest', 'nearest-up', 'previous', 'next'))
    def test_single_value(self, kind):
        f = interp1d([1.5], [6], kind=kind, bounds_error=False, fill_value=(2, 10))
        assert_array_equal(f([1, 1.5, 2]), [2, 6, 10])
        f = interp1d([1.5], [6], kind=kind, bounds_error=True)
        with assert_raises(ValueError, match='x_new is above'):
            f(2.0)