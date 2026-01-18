import numpy
import cupy
from cupy import _core
from cupyx import cusparse
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _csc
from cupyx.scipy.sparse import _csr
from cupyx.scipy.sparse import _data as sparse_data
from cupyx.scipy.sparse import _util
from cupyx.scipy.sparse import _sputils
def _with_data(self, data, copy=True):
    """Returns a matrix with the same sparsity structure as self,
        but with different data.  By default the index arrays
        (i.e. .row and .col) are copied.
        """
    if copy:
        return coo_matrix((data, (self.row.copy(), self.col.copy())), shape=self.shape, dtype=data.dtype)
    else:
        return coo_matrix((data, (self.row, self.col)), shape=self.shape, dtype=data.dtype)