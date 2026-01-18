import math
import llvmlite.ir
import numpy as np
from numba.core.extending import overload
from numba.core.imputils import impl_ret_untracked
from numba.core import typing, types, errors, lowering, cgutils
from numba.core.extending import register_jitable
from numba.np import npdatetime
from numba.cpython import cmathimpl, mathimpl, numbers
def _call_func_by_name_with_cast(context, builder, sig, args, func_name, ty=types.float64):
    mod = builder.module
    lty = context.get_argument_type(ty)
    fnty = llvmlite.ir.FunctionType(lty, [lty] * len(sig.args))
    fn = cgutils.insert_pure_function(mod, fnty, name=func_name)
    cast_args = [context.cast(builder, arg, argty, ty) for arg, argty in zip(args, sig.args)]
    result = builder.call(fn, cast_args)
    return context.cast(builder, result, types.float64, sig.return_type)