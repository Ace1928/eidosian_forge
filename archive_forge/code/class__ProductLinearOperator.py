import warnings
import cupy
from cupyx.scipy import sparse
from cupyx.scipy.sparse import _util
class _ProductLinearOperator(LinearOperator):

    def __init__(self, A, B):
        if not isinstance(A, LinearOperator) or not isinstance(B, LinearOperator):
            raise ValueError('both operands have to be a LinearOperator')
        if A.shape[1] != B.shape[0]:
            raise ValueError('cannot multiply %r and %r: shape mismatch' % (A, B))
        super(_ProductLinearOperator, self).__init__(_get_dtype([A, B]), (A.shape[0], B.shape[1]))
        self.args = (A, B)

    def _matvec(self, x):
        return self.args[0].matvec(self.args[1].matvec(x))

    def _rmatvec(self, x):
        return self.args[1].rmatvec(self.args[0].rmatvec(x))

    def _rmatmat(self, x):
        return self.args[1].rmatmat(self.args[0].rmatmat(x))

    def _matmat(self, x):
        return self.args[0].matmat(self.args[1].matmat(x))

    def _adjoint(self):
        A, B = self.args
        return B.H * A.H