import math
import numpy as np
import sys
import ctypes
import warnings
from collections import namedtuple
import llvmlite.binding as ll
from llvmlite import ir
from numba import literal_unroll
from numba.core.extending import (
from numba.core import errors
from numba.core import types, utils
from numba.core.unsafe.bytes import grab_byte, grab_uint64_t
from numba.cpython.randomimpl import (const_int, get_next_int, get_next_int32,
from ctypes import (  # noqa
@register_jitable(locals={'x': _Py_uhash_t, 'y': _Py_uhash_t, 'm': types.double, 'e': types.intc, 'sign': types.intc, '_PyHASH_MODULUS': _Py_uhash_t, '_PyHASH_BITS': types.intc})
def _Py_HashDouble(v):
    if not np.isfinite(v):
        if np.isinf(v):
            if v > 0:
                return _PyHASH_INF
            else:
                return -_PyHASH_INF
        elif _py310_or_later:
            x = _prng_random_hash()
            return process_return(x)
        else:
            return _PyHASH_NAN
    m, e = math.frexp(v)
    sign = 1
    if m < 0:
        sign = -1
        m = -m
    x = 0
    while m:
        x = x << 28 & _PyHASH_MODULUS | x >> _PyHASH_BITS - 28
        m *= 268435456.0
        e -= 28
        y = int(m)
        m -= y
        x += y
        if x >= _PyHASH_MODULUS:
            x -= _PyHASH_MODULUS
    if e >= 0:
        e = e % _PyHASH_BITS
    else:
        e = _PyHASH_BITS - 1 - (-1 - e) % _PyHASH_BITS
    x = x << e & _PyHASH_MODULUS | x >> _PyHASH_BITS - e
    x = x * sign
    return process_return(x)