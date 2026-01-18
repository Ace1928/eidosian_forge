from numpy import float32, float64, complex64, complex128, arange, array, \
from scipy.linalg import _fblas as fblas
from numpy.testing import assert_array_equal, \
import pytest
class BaseScal:
    """ Mixin class for scal testing """

    def test_simple(self):
        x = arange(3.0, dtype=self.dtype)
        real_x = x * 3.0
        x = self.blas_func(3.0, x)
        assert_array_equal(real_x, x)

    def test_x_stride(self):
        x = arange(6.0, dtype=self.dtype)
        real_x = x.copy()
        real_x[::2] = x[::2] * array(3.0, self.dtype)
        x = self.blas_func(3.0, x, n=3, incx=2)
        assert_array_equal(real_x, x)

    def test_x_bad_size(self):
        x = arange(12.0, dtype=self.dtype)
        with pytest.raises(Exception, match='failed for 1st keyword'):
            self.blas_func(2.0, x, n=4, incx=5)