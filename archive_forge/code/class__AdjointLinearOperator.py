import warnings
import cupy
from cupyx.scipy import sparse
from cupyx.scipy.sparse import _util
class _AdjointLinearOperator(LinearOperator):
    """Adjoint of arbitrary Linear Operator"""

    def __init__(self, A):
        shape = (A.shape[1], A.shape[0])
        super(_AdjointLinearOperator, self).__init__(dtype=A.dtype, shape=shape)
        self.A = A
        self.args = (A,)

    def _matvec(self, x):
        return self.A._rmatvec(x)

    def _rmatvec(self, x):
        return self.A._matvec(x)

    def _matmat(self, x):
        return self.A._rmatmat(x)

    def _rmatmat(self, x):
        return self.A._matmat(x)