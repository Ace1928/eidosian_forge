import warnings
import cupy
from cupyx.scipy import sparse
from cupyx.scipy.sparse import _util
class IdentityOperator(LinearOperator):

    def __init__(self, shape, dtype=None):
        super(IdentityOperator, self).__init__(dtype, shape)

    def _matvec(self, x):
        return x

    def _rmatvec(self, x):
        return x

    def _rmatmat(self, x):
        return x

    def _matmat(self, x):
        return x

    def _adjoint(self):
        return self