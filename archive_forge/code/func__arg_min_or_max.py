import cupy
import numpy as np
from cupy._core import internal
from cupy import _util
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _coo
from cupyx.scipy.sparse import _sputils
def _arg_min_or_max(self, axis, out, op, compare):
    if out is not None:
        raise ValueError("Sparse matrices do not support an 'out' parameter.")
    _sputils.validateaxis(axis)
    if axis is None:
        if 0 in self.shape:
            raise ValueError("Can't apply the operation to an empty matrix.")
        if self.nnz == 0:
            return 0
        else:
            zero = cupy.asarray(self.dtype.type(0))
            mat = self.tocoo()
            mat.sum_duplicates()
            am = op(mat.data)
            m = mat.data[am]
            return cupy.where(compare(m, zero), mat.row[am] * mat.shape[1] + mat.col[am], _non_zero_cmp(mat, am, zero, m))
    if axis < 0:
        axis += 2
    return self._arg_min_or_max_axis(axis, op)