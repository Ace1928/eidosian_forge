import cupy
from cupy import _core
from cupyx.scipy.sparse._base import isspmatrix
from cupyx.scipy.sparse._base import spmatrix
from cupy_backends.cuda.libs import cusparse
from cupy.cuda import device
from cupy.cuda import runtime
import numpy
def _set_arrayXarray_sparse(self, row, col, x):
    x = cupy.asarray(x.toarray(), dtype=self.dtype)
    x, _ = cupy.broadcast_arrays(x, row)
    self._set_arrayXarray(row, col, x)