from numpy import float32, float64, complex64, complex128, arange, array, \
from scipy.linalg import _fblas as fblas
from numpy.testing import assert_array_equal, \
import pytest
class BaseGemv:
    """ Mixin class for gemv tests """

    def get_data(self, x_stride=1, y_stride=1):
        mult = array(1, dtype=self.dtype)
        if self.dtype in [complex64, complex128]:
            mult = array(1 + 1j, dtype=self.dtype)
        from numpy.random import normal, seed
        seed(1234)
        alpha = array(1.0, dtype=self.dtype) * mult
        beta = array(1.0, dtype=self.dtype) * mult
        a = normal(0.0, 1.0, (3, 3)).astype(self.dtype) * mult
        x = arange(shape(a)[0] * x_stride, dtype=self.dtype) * mult
        y = arange(shape(a)[1] * y_stride, dtype=self.dtype) * mult
        return (alpha, beta, a, x, y)

    def test_simple(self):
        alpha, beta, a, x, y = self.get_data()
        desired_y = alpha * matrixmultiply(a, x) + beta * y
        y = self.blas_func(alpha, a, x, beta, y)
        assert_array_almost_equal(desired_y, y)

    def test_default_beta_y(self):
        alpha, beta, a, x, y = self.get_data()
        desired_y = matrixmultiply(a, x)
        y = self.blas_func(1, a, x)
        assert_array_almost_equal(desired_y, y)

    def test_simple_transpose(self):
        alpha, beta, a, x, y = self.get_data()
        desired_y = alpha * matrixmultiply(transpose(a), x) + beta * y
        y = self.blas_func(alpha, a, x, beta, y, trans=1)
        assert_array_almost_equal(desired_y, y)

    def test_simple_transpose_conj(self):
        alpha, beta, a, x, y = self.get_data()
        desired_y = alpha * matrixmultiply(transpose(conjugate(a)), x) + beta * y
        y = self.blas_func(alpha, a, x, beta, y, trans=2)
        assert_array_almost_equal(desired_y, y)

    def test_x_stride(self):
        alpha, beta, a, x, y = self.get_data(x_stride=2)
        desired_y = alpha * matrixmultiply(a, x[::2]) + beta * y
        y = self.blas_func(alpha, a, x, beta, y, incx=2)
        assert_array_almost_equal(desired_y, y)

    def test_x_stride_transpose(self):
        alpha, beta, a, x, y = self.get_data(x_stride=2)
        desired_y = alpha * matrixmultiply(transpose(a), x[::2]) + beta * y
        y = self.blas_func(alpha, a, x, beta, y, trans=1, incx=2)
        assert_array_almost_equal(desired_y, y)

    def test_x_stride_assert(self):
        alpha, beta, a, x, y = self.get_data(x_stride=2)
        with pytest.raises(Exception, match='failed for 3rd argument'):
            y = self.blas_func(1, a, x, 1, y, trans=0, incx=3)
        with pytest.raises(Exception, match='failed for 3rd argument'):
            y = self.blas_func(1, a, x, 1, y, trans=1, incx=3)

    def test_y_stride(self):
        alpha, beta, a, x, y = self.get_data(y_stride=2)
        desired_y = y.copy()
        desired_y[::2] = alpha * matrixmultiply(a, x) + beta * y[::2]
        y = self.blas_func(alpha, a, x, beta, y, incy=2)
        assert_array_almost_equal(desired_y, y)

    def test_y_stride_transpose(self):
        alpha, beta, a, x, y = self.get_data(y_stride=2)
        desired_y = y.copy()
        desired_y[::2] = alpha * matrixmultiply(transpose(a), x) + beta * y[::2]
        y = self.blas_func(alpha, a, x, beta, y, trans=1, incy=2)
        assert_array_almost_equal(desired_y, y)

    def test_y_stride_assert(self):
        alpha, beta, a, x, y = self.get_data(y_stride=2)
        with pytest.raises(Exception, match='failed for 2nd keyword'):
            y = self.blas_func(1, a, x, 1, y, trans=0, incy=3)
        with pytest.raises(Exception, match='failed for 2nd keyword'):
            y = self.blas_func(1, a, x, 1, y, trans=1, incy=3)