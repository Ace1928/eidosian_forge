import numpy as np
import operator
import llvmlite.ir
from llvmlite.ir import Constant
from numba.core import types, cgutils
from numba.core.cgutils import create_constant_array
from numba.core.imputils import (lower_builtin, lower_constant,
from numba.np import npdatetime_helpers, numpy_support, npyfuncs
from numba.extending import overload_method
from numba.core.config import IS_32BITS
from numba.core.errors import LoweringError
def _timedelta_times_number(context, builder, td_arg, td_type, number_arg, number_type, return_type):
    ret = alloc_timedelta_result(builder)
    with cgutils.if_likely(builder, is_not_nat(builder, td_arg)):
        if isinstance(number_type, types.Float):
            val = builder.sitofp(td_arg, number_arg.type)
            val = builder.fmul(val, number_arg)
            val = _cast_to_timedelta(context, builder, val)
        else:
            val = builder.mul(td_arg, number_arg)
        val = scale_timedelta(context, builder, val, td_type, return_type)
        builder.store(val, ret)
    return builder.load(ret)