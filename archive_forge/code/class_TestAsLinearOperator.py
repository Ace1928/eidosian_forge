from functools import partial
from itertools import product
import operator
from pytest import raises as assert_raises, warns
from numpy.testing import assert_, assert_equal
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg._interface as interface
from scipy.sparse._sputils import matrix
class TestAsLinearOperator:

    def setup_method(self):
        self.cases = []

        def make_cases(original, dtype):
            cases = []
            cases.append((matrix(original, dtype=dtype), original))
            cases.append((np.array(original, dtype=dtype), original))
            cases.append((sparse.csr_matrix(original, dtype=dtype), original))

            def mv(x, dtype):
                y = original.dot(x)
                if len(x.shape) == 2:
                    y = y.reshape(-1, 1)
                return y

            def rmv(x, dtype):
                return original.T.conj().dot(x)

            class BaseMatlike(interface.LinearOperator):
                args = ()

                def __init__(self, dtype):
                    self.dtype = np.dtype(dtype)
                    self.shape = original.shape

                def _matvec(self, x):
                    return mv(x, self.dtype)

            class HasRmatvec(BaseMatlike):
                args = ()

                def _rmatvec(self, x):
                    return rmv(x, self.dtype)

            class HasAdjoint(BaseMatlike):
                args = ()

                def _adjoint(self):
                    shape = (self.shape[1], self.shape[0])
                    matvec = partial(rmv, dtype=self.dtype)
                    rmatvec = partial(mv, dtype=self.dtype)
                    return interface.LinearOperator(matvec=matvec, rmatvec=rmatvec, dtype=self.dtype, shape=shape)

            class HasRmatmat(HasRmatvec):

                def _matmat(self, x):
                    return original.dot(x)

                def _rmatmat(self, x):
                    return original.T.conj().dot(x)
            cases.append((HasRmatvec(dtype), original))
            cases.append((HasAdjoint(dtype), original))
            cases.append((HasRmatmat(dtype), original))
            return cases
        original = np.array([[1, 2, 3], [4, 5, 6]])
        self.cases += make_cases(original, np.int32)
        self.cases += make_cases(original, np.float32)
        self.cases += make_cases(original, np.float64)
        self.cases += [(interface.aslinearoperator(M).T, A.T) for M, A in make_cases(original.T, np.float64)]
        self.cases += [(interface.aslinearoperator(M).H, A.T.conj()) for M, A in make_cases(original.T, np.float64)]
        original = np.array([[1, 2j, 3j], [4j, 5j, 6]])
        self.cases += make_cases(original, np.complex128)
        self.cases += [(interface.aslinearoperator(M).T, A.T) for M, A in make_cases(original.T, np.complex128)]
        self.cases += [(interface.aslinearoperator(M).H, A.T.conj()) for M, A in make_cases(original.T, np.complex128)]

    def test_basic(self):
        for M, A_array in self.cases:
            A = interface.aslinearoperator(M)
            M, N = A.shape
            xs = [np.array([1, 2, 3]), np.array([[1], [2], [3]])]
            ys = [np.array([1, 2]), np.array([[1], [2]])]
            if A.dtype == np.complex128:
                xs += [np.array([1, 2j, 3j]), np.array([[1], [2j], [3j]])]
                ys += [np.array([1, 2j]), np.array([[1], [2j]])]
            x2 = np.array([[1, 4], [2, 5], [3, 6]])
            for x in xs:
                assert_equal(A.matvec(x), A_array.dot(x))
                assert_equal(A * x, A_array.dot(x))
            assert_equal(A.matmat(x2), A_array.dot(x2))
            assert_equal(A * x2, A_array.dot(x2))
            for y in ys:
                assert_equal(A.rmatvec(y), A_array.T.conj().dot(y))
                assert_equal(A.T.matvec(y), A_array.T.dot(y))
                assert_equal(A.H.matvec(y), A_array.T.conj().dot(y))
            for y in ys:
                if y.ndim < 2:
                    continue
                assert_equal(A.rmatmat(y), A_array.T.conj().dot(y))
                assert_equal(A.T.matmat(y), A_array.T.dot(y))
                assert_equal(A.H.matmat(y), A_array.T.conj().dot(y))
            if hasattr(M, 'dtype'):
                assert_equal(A.dtype, M.dtype)
            assert_(hasattr(A, 'args'))

    def test_dot(self):
        for M, A_array in self.cases:
            A = interface.aslinearoperator(M)
            M, N = A.shape
            x0 = np.array([1, 2, 3])
            x1 = np.array([[1], [2], [3]])
            x2 = np.array([[1, 4], [2, 5], [3, 6]])
            assert_equal(A.dot(x0), A_array.dot(x0))
            assert_equal(A.dot(x1), A_array.dot(x1))
            assert_equal(A.dot(x2), A_array.dot(x2))