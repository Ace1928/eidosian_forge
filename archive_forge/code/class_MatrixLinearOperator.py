import warnings
import cupy
from cupyx.scipy import sparse
from cupyx.scipy.sparse import _util
class MatrixLinearOperator(LinearOperator):

    def __init__(self, A):
        super(MatrixLinearOperator, self).__init__(A.dtype, A.shape)
        self.A = A
        self.__adj = None
        self.args = (A,)

    def _matmat(self, X):
        return self.A.dot(X)

    def _adjoint(self):
        if self.__adj is None:
            self.__adj = _AdjointMatrixOperator(self)
        return self.__adj