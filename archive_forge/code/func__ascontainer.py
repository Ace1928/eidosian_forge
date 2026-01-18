from warnings import warn
import numpy as np
from scipy._lib._util import VisibleDeprecationWarning
from ._sputils import (asmatrix, check_reshape_kwargs, check_shape,
from ._matrix import spmatrix
@classmethod
def _ascontainer(cls, X, **kwargs):
    if issubclass(cls, sparray):
        return np.asarray(X, **kwargs)
    else:
        return asmatrix(X, **kwargs)