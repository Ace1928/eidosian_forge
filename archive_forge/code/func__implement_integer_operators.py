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
def _implement_integer_operators():
    ty = types.Integer
    lower_builtin(operator.add, ty, ty)(int_add_impl)
    lower_builtin(operator.iadd, ty, ty)(int_add_impl)
    lower_builtin(operator.sub, ty, ty)(int_sub_impl)
    lower_builtin(operator.isub, ty, ty)(int_sub_impl)
    lower_builtin(operator.mul, ty, ty)(int_mul_impl)
    lower_builtin(operator.imul, ty, ty)(int_mul_impl)
    lower_builtin(operator.eq, ty, ty)(int_eq_impl)
    lower_builtin(operator.ne, ty, ty)(int_ne_impl)
    lower_builtin(operator.lshift, ty, ty)(int_shl_impl)
    lower_builtin(operator.ilshift, ty, ty)(int_shl_impl)
    lower_builtin(operator.rshift, ty, ty)(int_shr_impl)
    lower_builtin(operator.irshift, ty, ty)(int_shr_impl)
    lower_builtin(operator.neg, ty)(int_negate_impl)
    lower_builtin(operator.pos, ty)(int_positive_impl)
    lower_builtin(operator.pow, ty, ty)(int_power_impl)
    lower_builtin(operator.ipow, ty, ty)(int_power_impl)
    lower_builtin(pow, ty, ty)(int_power_impl)
    for ty in types.unsigned_domain:
        lower_builtin(operator.lt, ty, ty)(int_ult_impl)
        lower_builtin(operator.le, ty, ty)(int_ule_impl)
        lower_builtin(operator.gt, ty, ty)(int_ugt_impl)
        lower_builtin(operator.ge, ty, ty)(int_uge_impl)
        lower_builtin(operator.pow, types.Float, ty)(int_power_impl)
        lower_builtin(operator.ipow, types.Float, ty)(int_power_impl)
        lower_builtin(pow, types.Float, ty)(int_power_impl)
        lower_builtin(abs, ty)(uint_abs_impl)
    lower_builtin(operator.lt, types.IntegerLiteral, types.IntegerLiteral)(int_slt_impl)
    lower_builtin(operator.gt, types.IntegerLiteral, types.IntegerLiteral)(int_slt_impl)
    lower_builtin(operator.le, types.IntegerLiteral, types.IntegerLiteral)(int_slt_impl)
    lower_builtin(operator.ge, types.IntegerLiteral, types.IntegerLiteral)(int_slt_impl)
    for ty in types.signed_domain:
        lower_builtin(operator.lt, ty, ty)(int_slt_impl)
        lower_builtin(operator.le, ty, ty)(int_sle_impl)
        lower_builtin(operator.gt, ty, ty)(int_sgt_impl)
        lower_builtin(operator.ge, ty, ty)(int_sge_impl)
        lower_builtin(operator.pow, types.Float, ty)(int_power_impl)
        lower_builtin(operator.ipow, types.Float, ty)(int_power_impl)
        lower_builtin(pow, types.Float, ty)(int_power_impl)
        lower_builtin(abs, ty)(int_abs_impl)