import functools
import numpy as _np
from . import numpy as mx_np  # pylint: disable=reimported
from .numpy.multiarray import _NUMPY_ARRAY_FUNCTION_DICT, _NUMPY_ARRAY_UFUNC_DICT
def _find_duplicate(strs):
    str_set = set()
    for s in strs:
        if s in str_set:
            return s
        else:
            str_set.add(s)
    return None