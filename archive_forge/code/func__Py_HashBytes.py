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
@register_jitable(locals={'_hash': _Py_uhash_t})
def _Py_HashBytes(val, _len):
    if _len == 0:
        return process_return(0)
    if _len < _Py_HASH_CUTOFF:
        _hash = _Py_uhash_t(5381)
        for idx in range(_len):
            _hash = (_hash << 5) + _hash + np.uint8(grab_byte(val, idx))
        _hash ^= _len
        _hash ^= _load_hashsecret('djbx33a_suffix')
    else:
        tmp = _siphasher(types.uint64(_load_hashsecret('siphash_k0')), types.uint64(_load_hashsecret('siphash_k1')), val, _len)
        _hash = process_return(tmp)
    return process_return(_hash)