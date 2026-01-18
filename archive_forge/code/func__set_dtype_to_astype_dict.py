import numpy
from cupy._core._dtype import get_dtype
import cupy
from cupy._core import _fusion_thread_local
from cupy._core import core
from cupy._core._scalar import get_typename
def _set_dtype_to_astype_dict():
    """Set a dict with dtypes and astype ufuncs to `_dtype_to_astype_dict`.

    Creates a ufunc for type cast operations, and set a dict with keys
    as the dtype of the output array and values as astype ufuncs.
    This function is called at most once.
    """
    global _dtype_to_astype_dict
    _dtype_to_astype_dict = {}
    dtype_list = [numpy.dtype(type_char) for type_char in '?bhilqBHILQefdFD']
    for t in dtype_list:
        name = 'astype_{}'.format(t)
        rules = tuple(['{}->{}'.format(s.char, t.char) for s in dtype_list])
        command = 'out0 = static_cast< {} >(in0)'.format(get_typename(t))
        _dtype_to_astype_dict[t] = core.create_ufunc(name, rules, command)