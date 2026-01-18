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
@overload_method(types.NumPyRandomGeneratorType, 'poisson')
def NumPyRandomGeneratorType_poisson(inst, lam, size=None):
    check_types(lam, [types.Float, types.Integer, int, float], 'lam')
    if isinstance(size, types.Omitted):
        size = size.value
    if is_nonelike(size):

        def impl(inst, lam, size=None):
            return np.int64(random_poisson(inst.bit_generator, lam))
        return impl
    else:
        check_size(size)

        def impl(inst, lam, size=None):
            out = np.empty(size, dtype=np.int64)
            out_f = out.flat
            for i in range(out.size):
                out_f[i] = random_poisson(inst.bit_generator, lam)
            return out
        return impl