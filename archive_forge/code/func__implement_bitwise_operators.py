import math
import numbers
import numpy as np
import operator
from llvmlite import ir
from llvmlite.ir import Constant
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core import typing, types, utils, errors, cgutils, optional
from numba.core.extending import intrinsic, overload_method
from numba.cpython.unsafe.numbers import viewer
def _implement_bitwise_operators():
    for ty in (types.Boolean, types.Integer):
        lower_builtin(operator.and_, ty, ty)(int_and_impl)
        lower_builtin(operator.iand, ty, ty)(int_and_impl)
        lower_builtin(operator.or_, ty, ty)(int_or_impl)
        lower_builtin(operator.ior, ty, ty)(int_or_impl)
        lower_builtin(operator.xor, ty, ty)(int_xor_impl)
        lower_builtin(operator.ixor, ty, ty)(int_xor_impl)
        lower_builtin(operator.invert, ty)(int_invert_impl)