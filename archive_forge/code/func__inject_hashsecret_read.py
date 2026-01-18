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
def _inject_hashsecret_read(tyctx, name):
    """Emit code to load the hashsecret.
    """
    if not isinstance(name, types.StringLiteral):
        raise errors.TypingError('requires literal string')
    sym = _hashsecret[name.literal_value].symbol
    resty = types.uint64
    sig = resty(name)

    def impl(cgctx, builder, sig, args):
        mod = builder.module
        try:
            gv = mod.get_global(sym)
        except KeyError:
            gv = ir.GlobalVariable(mod, ir.IntType(64), name=sym)
        v = builder.load(gv)
        return v
    return (sig, impl)