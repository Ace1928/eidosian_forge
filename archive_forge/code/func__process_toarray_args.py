from warnings import warn
import numpy as np
from scipy._lib._util import VisibleDeprecationWarning
from ._sputils import (asmatrix, check_reshape_kwargs, check_shape,
from ._matrix import spmatrix
def _process_toarray_args(self, order, out):
    if out is not None:
        if order is not None:
            raise ValueError('order cannot be specified if out is not None')
        if out.shape != self.shape or out.dtype != self.dtype:
            raise ValueError('out array must be same dtype and shape as sparse array')
        out[...] = 0.0
        return out
    else:
        return np.zeros(self.shape, dtype=self.dtype, order=order)