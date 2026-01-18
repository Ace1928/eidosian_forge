import math
import random
import numpy as np
from llvmlite import ir
from numba.core.cgutils import is_nonelike
from numba.core.extending import intrinsic, overload, register_jitable
from numba.core.imputils import (Registry, impl_ret_untracked,
from numba.core.typing import signature
from numba.core import types, cgutils
from numba.np import arrayobj
from numba.core.errors import NumbaTypeError
def _double_preprocessor(value):
    ty = ir.types.DoubleType()
    if isinstance(value, types.Integer):
        if value.signed:
            return lambda builder, v: builder.sitofp(v, ty)
        else:
            return lambda builder, v: builder.uitofp(v, ty)
    elif isinstance(value, types.Float):
        if value.bitwidth != 64:
            return lambda builder, v: builder.fpext(v, ty)
        else:
            return lambda _builder, v: v
    else:
        raise TypeError('Cannot convert {} to floating point type' % value)