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
@intrinsic
def _fpext(tyctx, val):

    def impl(cgctx, builder, signature, args):
        val = args[0]
        return builder.fpext(val, ir.DoubleType())
    sig = types.float64(types.float32)
    return (sig, impl)