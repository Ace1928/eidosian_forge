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
@overload_method(types.NumPyRandomGeneratorType, 'noncentral_f')
def NumPyRandomGeneratorType_noncentral_f(inst, dfnum, dfden, nonc, size=None):
    check_types(dfnum, [types.Float, types.Integer, int, float], 'dfnum')
    check_types(dfden, [types.Float, types.Integer, int, float], 'dfden')
    check_types(nonc, [types.Float, types.Integer, int, float], 'nonc')
    if isinstance(size, types.Omitted):
        size = size.value

    @register_jitable
    def check_arg_bounds(dfnum, dfden, nonc):
        if dfnum <= 0:
            raise ValueError('dfnum <= 0')
        if dfden <= 0:
            raise ValueError('dfden <= 0')
        if nonc < 0:
            raise ValueError('nonc < 0')
    if is_nonelike(size):

        def impl(inst, dfnum, dfden, nonc, size=None):
            check_arg_bounds(dfnum, dfden, nonc)
            return np.float64(random_noncentral_f(inst.bit_generator, dfnum, dfden, nonc))
        return impl
    else:
        check_size(size)

        def impl(inst, dfnum, dfden, nonc, size=None):
            check_arg_bounds(dfnum, dfden, nonc)
            out = np.empty(size, dtype=np.float64)
            out_f = out.flat
            for i in range(out.size):
                out_f[i] = random_noncentral_f(inst.bit_generator, dfnum, dfden, nonc)
            return out
        return impl