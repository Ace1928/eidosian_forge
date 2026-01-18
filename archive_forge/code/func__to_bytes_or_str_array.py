import functools
from .._utils import set_module
from .numerictypes import (
from .numeric import ndarray, compare_chararrays
from .numeric import array as narray
from numpy.core.multiarray import _vec_string
from numpy.core import overrides
from numpy.compat import asbytes
import numpy
def _to_bytes_or_str_array(result, output_dtype_like=None):
    """
    Helper function to cast a result back into an array
    with the appropriate dtype if an object array must be used
    as an intermediary.
    """
    ret = numpy.asarray(result.tolist())
    dtype = getattr(output_dtype_like, 'dtype', None)
    if dtype is not None:
        return ret.astype(type(dtype)(_get_num_chars(ret)), copy=False)
    return ret