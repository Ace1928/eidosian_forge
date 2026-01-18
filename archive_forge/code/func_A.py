import numpy
import cupy
from cupy import _core
from cupyx.scipy.sparse import _util
from cupyx.scipy.sparse import _sputils
@property
def A(self):
    """Dense ndarray representation of this matrix.

        This property is equivalent to
        :meth:`~cupyx.scipy.sparse.spmatrix.toarray` method.

        """
    return self.toarray()