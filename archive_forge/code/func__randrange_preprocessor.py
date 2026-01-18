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
def _randrange_preprocessor(bitwidth, ty):
    if ty.bitwidth != bitwidth:
        return ir.IRBuilder.sext if ty.signed else ir.IRBuilder.zext
    else:
        return lambda _builder, v, _ty: v