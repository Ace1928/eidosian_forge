import numpy as _np
from .blas import _get_funcs, _memoize_get_funcs
from scipy.linalg import _flapack
from re import compile as regex_compile
from scipy.linalg._flapack import *  # noqa: E402, F403
def _compute_lwork(routine, *args, **kwargs):
    """
    Round floating-point lwork returned by lapack to integer.

    Several LAPACK routines compute optimal values for LWORK, which
    they return in a floating-point variable. However, for large
    values of LWORK, single-precision floating point is not sufficient
    to hold the exact value --- some LAPACK versions (<= 3.5.0 at
    least) truncate the returned integer to single precision and in
    some cases this can be smaller than the required value.

    Examples
    --------
    >>> from scipy.linalg import lapack
    >>> n = 5000
    >>> s_r, s_lw = lapack.get_lapack_funcs(('sysvx', 'sysvx_lwork'))
    >>> lwork = lapack._compute_lwork(s_lw, n)
    >>> lwork
    32000

    """
    dtype = getattr(routine, 'dtype', None)
    int_dtype = getattr(routine, 'int_dtype', None)
    ret = routine(*args, **kwargs)
    if ret[-1] != 0:
        raise ValueError('Internal work array size computation failed: %d' % (ret[-1],))
    if len(ret) == 2:
        return _check_work_float(ret[0].real, dtype, int_dtype)
    else:
        return tuple((_check_work_float(x.real, dtype, int_dtype) for x in ret[:-1]))