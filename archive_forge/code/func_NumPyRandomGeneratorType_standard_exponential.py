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
@overload_method(types.NumPyRandomGeneratorType, 'standard_exponential')
def NumPyRandomGeneratorType_standard_exponential(inst, size=None, dtype=np.float64, method='zig'):
    check_types(method, [types.UnicodeType, str], 'method')
    dist_func_inv, nb_dt = _get_proper_func(random_standard_exponential_inv_f, random_standard_exponential_inv, dtype)
    dist_func, nb_dt = _get_proper_func(random_standard_exponential_f, random_standard_exponential, dtype)
    if isinstance(size, types.Omitted):
        size = size.value
    if is_nonelike(size):

        def impl(inst, size=None, dtype=np.float64, method='zig'):
            if method == 'zig':
                return nb_dt(dist_func(inst.bit_generator))
            elif method == 'inv':
                return nb_dt(dist_func_inv(inst.bit_generator))
            else:
                raise ValueError("Method must be either 'zig' or 'inv'")
        return impl
    else:
        check_size(size)

        def impl(inst, size=None, dtype=np.float64, method='zig'):
            out = np.empty(size, dtype=dtype)
            out_f = out.flat
            if method == 'zig':
                for i in range(out.size):
                    out_f[i] = dist_func(inst.bit_generator)
            elif method == 'inv':
                for i in range(out.size):
                    out_f[i] = dist_func_inv(inst.bit_generator)
            else:
                raise ValueError("Method must be either 'zig' or 'inv'")
            return out
        return impl