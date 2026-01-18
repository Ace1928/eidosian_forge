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
def _get_power_zerodiv_return(context, return_type):
    if isinstance(return_type, types.Integer) and (not context.error_model.raise_on_fp_zero_division):
        return -1 << return_type.bitwidth - 1
    else:
        return False