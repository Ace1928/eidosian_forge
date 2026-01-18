import warnings
import cupy
from cupyx.scipy import sparse
from cupyx.scipy.sparse import _util
class _ScaledLinearOperator(LinearOperator):

    def __init__(self, A, alpha):
        if not isinstance(A, LinearOperator):
            raise ValueError('LinearOperator expected as A')
        if not cupy.isscalar(alpha):
            raise ValueError('scalar expected as alpha')
        dtype = _get_dtype([A], [type(alpha)])
        super(_ScaledLinearOperator, self).__init__(dtype, A.shape)
        self.args = (A, alpha)

    def _matvec(self, x):
        return self.args[1] * self.args[0].matvec(x)

    def _rmatvec(self, x):
        return cupy.conj(self.args[1]) * self.args[0].rmatvec(x)

    def _rmatmat(self, x):
        return cupy.conj(self.args[1]) * self.args[0].rmatmat(x)

    def _matmat(self, x):
        return self.args[1] * self.args[0].matmat(x)

    def _adjoint(self):
        A, alpha = self.args
        return A.H * cupy.conj(alpha)