from numpy import float32, float64, complex64, complex128, arange, array, \
from scipy.linalg import _fblas as fblas
from numpy.testing import assert_array_equal, \
import pytest
class BaseSwap:
    """ Mixin class for swap tests """

    def test_simple(self):
        x = arange(3.0, dtype=self.dtype)
        y = zeros(shape(x), x.dtype)
        desired_x = y.copy()
        desired_y = x.copy()
        x, y = self.blas_func(x, y)
        assert_array_equal(desired_x, x)
        assert_array_equal(desired_y, y)

    def test_x_stride(self):
        x = arange(6.0, dtype=self.dtype)
        y = zeros(3, x.dtype)
        desired_x = y.copy()
        desired_y = x.copy()[::2]
        x, y = self.blas_func(x, y, n=3, incx=2)
        assert_array_equal(desired_x, x[::2])
        assert_array_equal(desired_y, y)

    def test_y_stride(self):
        x = arange(3.0, dtype=self.dtype)
        y = zeros(6, x.dtype)
        desired_x = y.copy()[::2]
        desired_y = x.copy()
        x, y = self.blas_func(x, y, n=3, incy=2)
        assert_array_equal(desired_x, x)
        assert_array_equal(desired_y, y[::2])

    def test_x_and_y_stride(self):
        x = arange(12.0, dtype=self.dtype)
        y = zeros(6, x.dtype)
        desired_x = y.copy()[::2]
        desired_y = x.copy()[::4]
        x, y = self.blas_func(x, y, n=3, incx=4, incy=2)
        assert_array_equal(desired_x, x[::4])
        assert_array_equal(desired_y, y[::2])

    def test_x_bad_size(self):
        x = arange(12.0, dtype=self.dtype)
        y = zeros(6, x.dtype)
        with pytest.raises(Exception, match='failed for 1st keyword'):
            self.blas_func(x, y, n=4, incx=5)

    def test_y_bad_size(self):
        x = arange(12.0, dtype=self.dtype)
        y = zeros(6, x.dtype)
        with pytest.raises(Exception, match='failed for 1st keyword'):
            self.blas_func(x, y, n=3, incy=5)