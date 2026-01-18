import os
from numpy import (
from numpy.core.multiarray import _flagdict, flagsobj
def _ctype_from_dtype(dtype):
    if dtype.fields is not None:
        return _ctype_from_dtype_structured(dtype)
    elif dtype.subdtype is not None:
        return _ctype_from_dtype_subarray(dtype)
    else:
        return _ctype_from_dtype_scalar(dtype)