import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.polynomial import polyutils as pu
from numba import literal_unroll
from numba.core import types, errors
from numba.core.extending import overload
from numba.np.numpy_support import type_can_asarray, as_dtype, from_dtype
def _poly_result_dtype(*args):
    res_dtype = np.float64
    for item in args:
        if isinstance(item, types.BaseTuple):
            s1 = item.types
        elif isinstance(item, types.List):
            s1 = [_get_list_type(item)]
        elif isinstance(item, types.Number):
            s1 = [item]
        elif isinstance(item, types.Array):
            s1 = [item.dtype]
        else:
            msg = 'Input dtype must be scalar'
            raise errors.TypingError(msg)
        try:
            l = [as_dtype(t) for t in s1]
            l.append(res_dtype)
            res_dtype = np.result_type(*l)
        except errors.NumbaNotImplementedError:
            msg = 'Input dtype must be scalar.'
            raise errors.TypingError(msg)
    return from_dtype(res_dtype)