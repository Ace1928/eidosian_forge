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
@register_jitable(locals={'v0': types.uint64, 'v1': types.uint64, 'v2': types.uint64, 'v3': types.uint64})
def _SINGLE_ROUND(v0, v1, v2, v3):
    v0, v1, v2, v3 = _HALF_ROUND(v0, v1, v2, v3, 13, 16)
    v2, v1, v0, v3 = _HALF_ROUND(v2, v1, v0, v3, 17, 21)
    return (v0, v1, v2, v3)