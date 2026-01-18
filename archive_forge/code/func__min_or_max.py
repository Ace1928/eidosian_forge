import cupy
import numpy as np
from cupy._core import internal
from cupy import _util
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _coo
from cupyx.scipy.sparse import _sputils
def _min_or_max(self, axis, out, min_or_max, explicit):
    if out is not None:
        raise ValueError("Sparse matrices do not support an 'out' parameter.")
    _sputils.validateaxis(axis)
    if axis is None:
        if 0 in self.shape:
            raise ValueError('zero-size array to reduction operation')
        zero = cupy.zeros((), dtype=self.dtype)
        if self.nnz == 0:
            return zero
        self.sum_duplicates()
        m = min_or_max(self.data)
        if explicit:
            return m
        if self.nnz != internal.prod(self.shape):
            if min_or_max is cupy.min:
                m = cupy.minimum(zero, m)
            elif min_or_max is cupy.max:
                m = cupy.maximum(zero, m)
            else:
                assert False
        return m
    if axis < 0:
        axis += 2
    return self._min_or_max_axis(axis, min_or_max, explicit)