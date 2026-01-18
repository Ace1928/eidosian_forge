from functools import partial
from itertools import product
import operator
from pytest import raises as assert_raises, warns
from numpy.testing import assert_, assert_equal
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg._interface as interface
from scipy.sparse._sputils import matrix
class TestLinearOperator:

    def setup_method(self):
        self.A = np.array([[1, 2, 3], [4, 5, 6]])
        self.B = np.array([[1, 2], [3, 4], [5, 6]])
        self.C = np.array([[1, 2], [3, 4]])

    def test_matvec(self):

        def get_matvecs(A):
            return [{'shape': A.shape, 'matvec': lambda x: np.dot(A, x).reshape(A.shape[0]), 'rmatvec': lambda x: np.dot(A.T.conj(), x).reshape(A.shape[1])}, {'shape': A.shape, 'matvec': lambda x: np.dot(A, x), 'rmatvec': lambda x: np.dot(A.T.conj(), x), 'rmatmat': lambda x: np.dot(A.T.conj(), x), 'matmat': lambda x: np.dot(A, x)}]
        for matvecs in get_matvecs(self.A):
            A = interface.LinearOperator(**matvecs)
            assert_(A.args == ())
            assert_equal(A.matvec(np.array([1, 2, 3])), [14, 32])
            assert_equal(A.matvec(np.array([[1], [2], [3]])), [[14], [32]])
            assert_equal(A * np.array([1, 2, 3]), [14, 32])
            assert_equal(A * np.array([[1], [2], [3]]), [[14], [32]])
            assert_equal(A.dot(np.array([1, 2, 3])), [14, 32])
            assert_equal(A.dot(np.array([[1], [2], [3]])), [[14], [32]])
            assert_equal(A.matvec(matrix([[1], [2], [3]])), [[14], [32]])
            assert_equal(A * matrix([[1], [2], [3]]), [[14], [32]])
            assert_equal(A.dot(matrix([[1], [2], [3]])), [[14], [32]])
            assert_equal(2 * A * [1, 1, 1], [12, 30])
            assert_equal((2 * A).rmatvec([1, 1]), [10, 14, 18])
            assert_equal((2 * A).H.matvec([1, 1]), [10, 14, 18])
            assert_equal(2 * A * [[1], [1], [1]], [[12], [30]])
            assert_equal((2 * A).matmat([[1], [1], [1]]), [[12], [30]])
            assert_equal(A * 2 * [1, 1, 1], [12, 30])
            assert_equal(A * 2 * [[1], [1], [1]], [[12], [30]])
            assert_equal(2j * A * [1, 1, 1], [12j, 30j])
            assert_equal((A + A) * [1, 1, 1], [12, 30])
            assert_equal((A + A).rmatvec([1, 1]), [10, 14, 18])
            assert_equal((A + A).H.matvec([1, 1]), [10, 14, 18])
            assert_equal((A + A) * [[1], [1], [1]], [[12], [30]])
            assert_equal((A + A).matmat([[1], [1], [1]]), [[12], [30]])
            assert_equal(-A * [1, 1, 1], [-6, -15])
            assert_equal(-A * [[1], [1], [1]], [[-6], [-15]])
            assert_equal((A - A) * [1, 1, 1], [0, 0])
            assert_equal((A - A) * [[1], [1], [1]], [[0], [0]])
            X = np.array([[1, 2], [3, 4]])
            assert_equal((2 * A).rmatmat(X), np.dot((2 * self.A).T, X))
            assert_equal((A * 2).rmatmat(X), np.dot((self.A * 2).T, X))
            assert_equal((2j * A).rmatmat(X), np.dot((2j * self.A).T.conj(), X))
            assert_equal((A * 2j).rmatmat(X), np.dot((self.A * 2j).T.conj(), X))
            assert_equal((A + A).rmatmat(X), np.dot((self.A + self.A).T, X))
            assert_equal((A + 2j * A).rmatmat(X), np.dot((self.A + 2j * self.A).T.conj(), X))
            assert_equal((-A).rmatmat(X), np.dot((-self.A).T, X))
            assert_equal((A - A).rmatmat(X), np.dot((self.A - self.A).T, X))
            assert_equal((2j * A).rmatmat(2j * X), np.dot((2j * self.A).T.conj(), 2j * X))
            z = A + A
            assert_(len(z.args) == 2 and z.args[0] is A and (z.args[1] is A))
            z = 2 * A
            assert_(len(z.args) == 2 and z.args[0] is A and (z.args[1] == 2))
            assert_(isinstance(A.matvec([1, 2, 3]), np.ndarray))
            assert_(isinstance(A.matvec(np.array([[1], [2], [3]])), np.ndarray))
            assert_(isinstance(A * np.array([1, 2, 3]), np.ndarray))
            assert_(isinstance(A * np.array([[1], [2], [3]]), np.ndarray))
            assert_(isinstance(A.dot(np.array([1, 2, 3])), np.ndarray))
            assert_(isinstance(A.dot(np.array([[1], [2], [3]])), np.ndarray))
            assert_(isinstance(A.matvec(matrix([[1], [2], [3]])), np.ndarray))
            assert_(isinstance(A * matrix([[1], [2], [3]]), np.ndarray))
            assert_(isinstance(A.dot(matrix([[1], [2], [3]])), np.ndarray))
            assert_(isinstance(2 * A, interface._ScaledLinearOperator))
            assert_(isinstance(2j * A, interface._ScaledLinearOperator))
            assert_(isinstance(A + A, interface._SumLinearOperator))
            assert_(isinstance(-A, interface._ScaledLinearOperator))
            assert_(isinstance(A - A, interface._SumLinearOperator))
            assert_(isinstance(A / 2, interface._ScaledLinearOperator))
            assert_(isinstance(A / 2j, interface._ScaledLinearOperator))
            assert_((A * 3 / 3).args[0] is A)
            result = A @ np.array([1, 2, 3])
            B = A * 3
            C = A / 5
            assert_equal(A @ np.array([1, 2, 3]), result)
            assert_((2j * A).dtype == np.complex128)
            msg = 'Can only divide a linear operator by a scalar.'
            with assert_raises(ValueError, match=msg):
                A / np.array([1, 2])
            assert_raises(ValueError, A.matvec, np.array([1, 2]))
            assert_raises(ValueError, A.matvec, np.array([1, 2, 3, 4]))
            assert_raises(ValueError, A.matvec, np.array([[1], [2]]))
            assert_raises(ValueError, A.matvec, np.array([[1], [2], [3], [4]]))
            assert_raises(ValueError, lambda: A * A)
            assert_raises(ValueError, lambda: A ** 2)
        for matvecsA, matvecsB in product(get_matvecs(self.A), get_matvecs(self.B)):
            A = interface.LinearOperator(**matvecsA)
            B = interface.LinearOperator(**matvecsB)
            AtimesB = self.A.dot(self.B)
            X = np.array([[1, 2], [3, 4]])
            assert_equal((A * B).rmatmat(X), np.dot(AtimesB.T, X))
            assert_equal((2j * A * B).rmatmat(X), np.dot((2j * AtimesB).T.conj(), X))
            assert_equal(A * B * [1, 1], [50, 113])
            assert_equal(A * B * [[1], [1]], [[50], [113]])
            assert_equal((A * B).matmat([[1], [1]]), [[50], [113]])
            assert_equal((A * B).rmatvec([1, 1]), [71, 92])
            assert_equal((A * B).H.matvec([1, 1]), [71, 92])
            assert_(isinstance(A * B, interface._ProductLinearOperator))
            assert_raises(ValueError, lambda: A + B)
            assert_raises(ValueError, lambda: A ** 2)
            z = A * B
            assert_(len(z.args) == 2 and z.args[0] is A and (z.args[1] is B))
        for matvecsC in get_matvecs(self.C):
            C = interface.LinearOperator(**matvecsC)
            X = np.array([[1, 2], [3, 4]])
            assert_equal(C.rmatmat(X), np.dot(self.C.T, X))
            assert_equal((C ** 2).rmatmat(X), np.dot(np.dot(self.C, self.C).T, X))
            assert_equal(C ** 2 * [1, 1], [17, 37])
            assert_equal((C ** 2).rmatvec([1, 1]), [22, 32])
            assert_equal((C ** 2).H.matvec([1, 1]), [22, 32])
            assert_equal((C ** 2).matmat([[1], [1]]), [[17], [37]])
            assert_(isinstance(C ** 2, interface._PowerLinearOperator))

    def test_matmul(self):
        D = {'shape': self.A.shape, 'matvec': lambda x: np.dot(self.A, x).reshape(self.A.shape[0]), 'rmatvec': lambda x: np.dot(self.A.T.conj(), x).reshape(self.A.shape[1]), 'rmatmat': lambda x: np.dot(self.A.T.conj(), x), 'matmat': lambda x: np.dot(self.A, x)}
        A = interface.LinearOperator(**D)
        B = np.array([[1 + 1j, 2, 3], [4, 5, 6], [7, 8, 9]])
        b = B[0]
        assert_equal(operator.matmul(A, b), A * b)
        assert_equal(operator.matmul(A, b.reshape(-1, 1)), A * b.reshape(-1, 1))
        assert_equal(operator.matmul(A, B), A * B)
        assert_equal(operator.matmul(b, A.H), b * A.H)
        assert_equal(operator.matmul(b.reshape(1, -1), A.H), b.reshape(1, -1) * A.H)
        assert_equal(operator.matmul(B, A.H), B * A.H)
        assert_raises(ValueError, operator.matmul, A, 2)
        assert_raises(ValueError, operator.matmul, 2, A)