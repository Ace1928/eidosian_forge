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
@overload_method(types.NumPyRandomGeneratorType, 'standard_gamma')
def NumPyRandomGeneratorType_standard_gamma(inst, shape, size=None, dtype=np.float64):
    check_types(shape, [types.Float, types.Integer, int, float], 'shape')
    dist_func, nb_dt = _get_proper_func(random_standard_gamma_f, random_standard_gamma, dtype)
    if isinstance(size, types.Omitted):
        size = size.value
    if is_nonelike(size):

        def impl(inst, shape, size=None, dtype=np.float64):
            return nb_dt(dist_func(inst.bit_generator, shape))
        return impl
    else:
        check_size(size)

        def impl(inst, shape, size=None, dtype=np.float64):
            out = np.empty(size, dtype=dtype)
            out_f = out.flat
            for i in range(out.size):
                out_f[i] = dist_func(inst.bit_generator, shape)
            return out
        return impl