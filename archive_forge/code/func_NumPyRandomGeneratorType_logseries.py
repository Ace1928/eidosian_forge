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
@overload_method(types.NumPyRandomGeneratorType, 'logseries')
def NumPyRandomGeneratorType_logseries(inst, p, size=None):
    check_types(p, [types.Float, types.Integer, int, float], 'p')
    if isinstance(size, types.Omitted):
        size = size.value

    @register_jitable
    def check_arg_bounds(p):
        if p < 0 or p >= 1 or np.isnan(p):
            raise ValueError('p < 0, p >= 1 or p is NaN')
    if is_nonelike(size):

        def impl(inst, p, size=None):
            check_arg_bounds(p)
            return np.int64(random_logseries(inst.bit_generator, p))
        return impl
    else:
        check_size(size)

        def impl(inst, p, size=None):
            check_arg_bounds(p)
            out = np.empty(size, dtype=np.int64)
            out_f = out.flat
            for i in range(out.size):
                out_f[i] = random_logseries(inst.bit_generator, p)
            return out
        return impl