import collections.abc
import numpy as np
from numpy import matrix, asmatrix, bmat
from numpy.testing import (
from numpy.linalg import matrix_power
from numpy.matrixlib import mat
class TestAlgebra:

    def test_basic(self):
        import numpy.linalg as linalg
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        mA = matrix(A)
        B = np.identity(2)
        for i in range(6):
            assert_(np.allclose((mA ** i).A, B))
            B = np.dot(B, A)
        Ainv = linalg.inv(A)
        B = np.identity(2)
        for i in range(6):
            assert_(np.allclose((mA ** (-i)).A, B))
            B = np.dot(B, Ainv)
        assert_(np.allclose((mA * mA).A, np.dot(A, A)))
        assert_(np.allclose((mA + mA).A, A + A))
        assert_(np.allclose((3 * mA).A, 3 * A))
        mA2 = matrix(A)
        mA2 *= 3
        assert_(np.allclose(mA2.A, 3 * A))

    def test_pow(self):
        """Test raising a matrix to an integer power works as expected."""
        m = matrix('1. 2.; 3. 4.')
        m2 = m.copy()
        m2 **= 2
        mi = m.copy()
        mi **= -1
        m4 = m2.copy()
        m4 **= 2
        assert_array_almost_equal(m2, m ** 2)
        assert_array_almost_equal(m4, np.dot(m2, m2))
        assert_array_almost_equal(np.dot(mi, m), np.eye(2))

    def test_scalar_type_pow(self):
        m = matrix([[1, 2], [3, 4]])
        for scalar_t in [np.int8, np.uint8]:
            two = scalar_t(2)
            assert_array_almost_equal(m ** 2, m ** two)

    def test_notimplemented(self):
        """Check that 'not implemented' operations produce a failure."""
        A = matrix([[1.0, 2.0], [3.0, 4.0]])
        with assert_raises(TypeError):
            1.0 ** A
        with assert_raises(TypeError):
            A * object()