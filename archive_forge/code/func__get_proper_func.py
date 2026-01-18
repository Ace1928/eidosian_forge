import numpy as np
from numba.core import types
from numba.core.extending import overload_method, register_jitable
from numba.np.numpy_support import as_dtype, from_dtype
from numba.np.random.generator_core import next_float, next_double
from numba.np.numpy_support import is_nonelike
from numba.core.errors import TypingError
from numba.core.types.containers import Tuple, UniTuple
from numba.np.random.distributions import \
from numba.np.random import random_methods
def _get_proper_func(func_32, func_64, dtype, dist_name='the given'):
    """
        Most of the standard NumPy distributions that accept dtype argument
        only support either np.float32 or np.float64 as dtypes.

        This is a helper function that helps Numba select the proper underlying
        implementation according to provided dtype.
    """
    if isinstance(dtype, types.Omitted):
        dtype = dtype.value
    np_dt = dtype
    if isinstance(dtype, type):
        nb_dt = from_dtype(np.dtype(dtype))
    elif isinstance(dtype, types.NumberClass):
        nb_dt = dtype
        np_dt = as_dtype(nb_dt)
    if np_dt not in [np.float32, np.float64]:
        raise TypingError('Argument dtype is not one of the' + ' expected type(s): ' + ' np.float32 or np.float64')
    if np_dt == np.float32:
        next_func = func_32
    else:
        next_func = func_64
    return (next_func, nb_dt)