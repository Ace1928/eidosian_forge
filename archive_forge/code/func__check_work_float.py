import numpy as _np
from .blas import _get_funcs, _memoize_get_funcs
from scipy.linalg import _flapack
from re import compile as regex_compile
from scipy.linalg._flapack import *  # noqa: E402, F403
def _check_work_float(value, dtype, int_dtype):
    """
    Convert LAPACK-returned work array size float to integer,
    carefully for single-precision types.
    """
    if dtype == _np.float32 or dtype == _np.complex64:
        value = _np.nextafter(value, _np.inf, dtype=_np.float32)
    value = int(value)
    if int_dtype.itemsize == 4:
        if value < 0 or value > _int32_max:
            raise ValueError('Too large work array required -- computation cannot be performed with standard 32-bit LAPACK.')
    elif int_dtype.itemsize == 8:
        if value < 0 or value > _int64_max:
            raise ValueError('Too large work array required -- computation cannot be performed with standard 64-bit LAPACK.')
    return value