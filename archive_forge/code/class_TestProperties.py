import collections.abc
import numpy as np
from numpy import matrix, asmatrix, bmat
from numpy.testing import (
from numpy.linalg import matrix_power
from numpy.matrixlib import mat
class TestProperties:

    def test_sum(self):
        """Test whether matrix.sum(axis=1) preserves orientation.
        Fails in NumPy <= 0.9.6.2127.
        """
        M = matrix([[1, 2, 0, 0], [3, 4, 0, 0], [1, 2, 1, 2], [3, 4, 3, 4]])
        sum0 = matrix([8, 12, 4, 6])
        sum1 = matrix([3, 7, 6, 14]).T
        sumall = 30
        assert_array_equal(sum0, M.sum(axis=0))
        assert_array_equal(sum1, M.sum(axis=1))
        assert_equal(sumall, M.sum())
        assert_array_equal(sum0, np.sum(M, axis=0))
        assert_array_equal(sum1, np.sum(M, axis=1))
        assert_equal(sumall, np.sum(M))

    def test_prod(self):
        x = matrix([[1, 2, 3], [4, 5, 6]])
        assert_equal(x.prod(), 720)
        assert_equal(x.prod(0), matrix([[4, 10, 18]]))
        assert_equal(x.prod(1), matrix([[6], [120]]))
        assert_equal(np.prod(x), 720)
        assert_equal(np.prod(x, axis=0), matrix([[4, 10, 18]]))
        assert_equal(np.prod(x, axis=1), matrix([[6], [120]]))
        y = matrix([0, 1, 3])
        assert_(y.prod() == 0)

    def test_max(self):
        x = matrix([[1, 2, 3], [4, 5, 6]])
        assert_equal(x.max(), 6)
        assert_equal(x.max(0), matrix([[4, 5, 6]]))
        assert_equal(x.max(1), matrix([[3], [6]]))
        assert_equal(np.max(x), 6)
        assert_equal(np.max(x, axis=0), matrix([[4, 5, 6]]))
        assert_equal(np.max(x, axis=1), matrix([[3], [6]]))

    def test_min(self):
        x = matrix([[1, 2, 3], [4, 5, 6]])
        assert_equal(x.min(), 1)
        assert_equal(x.min(0), matrix([[1, 2, 3]]))
        assert_equal(x.min(1), matrix([[1], [4]]))
        assert_equal(np.min(x), 1)
        assert_equal(np.min(x, axis=0), matrix([[1, 2, 3]]))
        assert_equal(np.min(x, axis=1), matrix([[1], [4]]))

    def test_ptp(self):
        x = np.arange(4).reshape((2, 2))
        assert_(x.ptp() == 3)
        assert_(np.all(x.ptp(0) == np.array([2, 2])))
        assert_(np.all(x.ptp(1) == np.array([1, 1])))

    def test_var(self):
        x = np.arange(9).reshape((3, 3))
        mx = x.view(np.matrix)
        assert_equal(x.var(ddof=0), mx.var(ddof=0))
        assert_equal(x.var(ddof=1), mx.var(ddof=1))

    def test_basic(self):
        import numpy.linalg as linalg
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        mA = matrix(A)
        assert_(np.allclose(linalg.inv(A), mA.I))
        assert_(np.all(np.array(np.transpose(A) == mA.T)))
        assert_(np.all(np.array(np.transpose(A) == mA.H)))
        assert_(np.all(A == mA.A))
        B = A + 2j * A
        mB = matrix(B)
        assert_(np.allclose(linalg.inv(B), mB.I))
        assert_(np.all(np.array(np.transpose(B) == mB.T)))
        assert_(np.all(np.array(np.transpose(B).conj() == mB.H)))

    def test_pinv(self):
        x = matrix(np.arange(6).reshape(2, 3))
        xpinv = matrix([[-0.77777778, 0.27777778], [-0.11111111, 0.11111111], [0.55555556, -0.05555556]])
        assert_almost_equal(x.I, xpinv)

    def test_comparisons(self):
        A = np.arange(100).reshape(10, 10)
        mA = matrix(A)
        mB = matrix(A) + 0.1
        assert_(np.all(mB == A + 0.1))
        assert_(np.all(mB == matrix(A + 0.1)))
        assert_(not np.any(mB == matrix(A - 0.1)))
        assert_(np.all(mA < mB))
        assert_(np.all(mA <= mB))
        assert_(np.all(mA <= mA))
        assert_(not np.any(mA < mA))
        assert_(not np.any(mB < mA))
        assert_(np.all(mB >= mA))
        assert_(np.all(mB >= mB))
        assert_(not np.any(mB > mB))
        assert_(np.all(mA == mA))
        assert_(not np.any(mA == mB))
        assert_(np.all(mB != mA))
        assert_(not np.all(abs(mA) > 0))
        assert_(np.all(abs(mB > 0)))

    def test_asmatrix(self):
        A = np.arange(100).reshape(10, 10)
        mA = asmatrix(A)
        A[0, 0] = -10
        assert_(A[0, 0] == mA[0, 0])

    def test_noaxis(self):
        A = matrix([[1, 0], [0, 1]])
        assert_(A.sum() == matrix(2))
        assert_(A.mean() == matrix(0.5))

    def test_repr(self):
        A = matrix([[1, 0], [0, 1]])
        assert_(repr(A) == 'matrix([[1, 0],\n        [0, 1]])')

    def test_make_bool_matrix_from_str(self):
        A = matrix('True; True; False')
        B = matrix([[True], [True], [False]])
        assert_array_equal(A, B)