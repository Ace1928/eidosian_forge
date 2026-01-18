import functools
import numpy as _np
from . import numpy as mx_np  # pylint: disable=reimported
from .numpy.multiarray import _NUMPY_ARRAY_FUNCTION_DICT, _NUMPY_ARRAY_UFUNC_DICT
@functools.wraps(func)
def _run_with_array_func_proto(*args, **kwargs):
    if cur_np_ver >= np_1_17_ver:
        try:
            func(*args, **kwargs)
        except Exception as e:
            raise RuntimeError('Running function {} with NumPy array function protocol failed with exception {}'.format(func.__name__, str(e)))