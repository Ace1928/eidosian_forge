import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
class Test_DerivedBivariateSpline:
    """Test the creation, usage, and attribute access of the (private)
    _DerivedBivariateSpline class.
    """

    def setup_method(self):
        x = np.concatenate(list(zip(range(10), range(10))))
        y = np.concatenate(list(zip(range(10), range(1, 11))))
        z = np.concatenate((np.linspace(3, 1, 10), np.linspace(1, 3, 10)))
        with suppress_warnings() as sup:
            sup.record(UserWarning, '\nThe coefficients of the spline')
            self.lut_lsq = LSQBivariateSpline(x, y, z, linspace(0.5, 19.5, 4), linspace(1.5, 20.5, 4), eps=0.01)
        self.lut_smooth = SmoothBivariateSpline(x, y, z)
        xx = linspace(0, 1, 20)
        yy = xx + 1.0
        zz = array([np.roll(z, i) for i in range(z.size)])
        self.lut_rect = RectBivariateSpline(xx, yy, zz)
        self.orders = list(itertools.product(range(3), range(3)))

    def test_creation_from_LSQ(self):
        for nux, nuy in self.orders:
            lut_der = self.lut_lsq.partial_derivative(nux, nuy)
            a = lut_der(3.5, 3.5, grid=False)
            b = self.lut_lsq(3.5, 3.5, dx=nux, dy=nuy, grid=False)
            assert_equal(a, b)

    def test_creation_from_Smooth(self):
        for nux, nuy in self.orders:
            lut_der = self.lut_smooth.partial_derivative(nux, nuy)
            a = lut_der(5.5, 5.5, grid=False)
            b = self.lut_smooth(5.5, 5.5, dx=nux, dy=nuy, grid=False)
            assert_equal(a, b)

    def test_creation_from_Rect(self):
        for nux, nuy in self.orders:
            lut_der = self.lut_rect.partial_derivative(nux, nuy)
            a = lut_der(0.5, 1.5, grid=False)
            b = self.lut_rect(0.5, 1.5, dx=nux, dy=nuy, grid=False)
            assert_equal(a, b)

    def test_invalid_attribute_fp(self):
        der = self.lut_rect.partial_derivative(1, 1)
        with assert_raises(AttributeError):
            der.fp

    def test_invalid_attribute_get_residual(self):
        der = self.lut_smooth.partial_derivative(1, 1)
        with assert_raises(AttributeError):
            der.get_residual()