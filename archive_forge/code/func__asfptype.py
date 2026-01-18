from warnings import warn
import numpy as np
from scipy._lib._util import VisibleDeprecationWarning
from ._sputils import (asmatrix, check_reshape_kwargs, check_shape,
from ._matrix import spmatrix
def _asfptype(self):
    """Upcast array to a floating point format (if necessary)"""
    fp_types = ['f', 'd', 'F', 'D']
    if self.dtype.char in fp_types:
        return self
    else:
        for fp_type in fp_types:
            if self.dtype <= np.dtype(fp_type):
                return self.astype(fp_type)
        raise TypeError('cannot upcast [%s] to a floating point format' % self.dtype.name)