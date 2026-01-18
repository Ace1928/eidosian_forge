import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
import pytest
import scipy.linalg
import scipy.sparse.linalg
from scipy.sparse.linalg._onenormest import _onenormest_core, _algorithm_2_2
class MatrixProductOperator(scipy.sparse.linalg.LinearOperator):
    """
    This is purely for onenormest testing.
    """

    def __init__(self, A, B):
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError('expected ndarrays representing matrices')
        if A.shape[1] != B.shape[0]:
            raise ValueError('incompatible shapes')
        self.A = A
        self.B = B
        self.ndim = 2
        self.shape = (A.shape[0], B.shape[1])

    def _matvec(self, x):
        return np.dot(self.A, np.dot(self.B, x))

    def _rmatvec(self, x):
        return np.dot(np.dot(x, self.A), self.B)

    def _matmat(self, X):
        return np.dot(self.A, np.dot(self.B, X))

    @property
    def T(self):
        return MatrixProductOperator(self.B.T, self.A.T)