import warnings
import numpy as np
from numpy import asarray_chkfinite
from ._misc import LinAlgError, _datacopied, LinAlgWarning
from .lapack import get_lapack_funcs
def _select_function(sort):
    if callable(sort):
        sfunction = sort
    elif sort == 'lhp':
        sfunction = _lhp
    elif sort == 'rhp':
        sfunction = _rhp
    elif sort == 'iuc':
        sfunction = _iuc
    elif sort == 'ouc':
        sfunction = _ouc
    else:
        raise ValueError("sort parameter must be None, a callable, or one of ('lhp','rhp','iuc','ouc')")
    return sfunction