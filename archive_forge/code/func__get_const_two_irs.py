import numbers
import copy
import types as pytypes
from operator import add
import operator
import numpy as np
import numba.parfors.parfor
from numba.core import types, ir, rewrites, config, ir_utils
from numba.core.typing.templates import infer_global, AbstractTemplate
from numba.core.typing import signature
from numba.core import  utils, typing
from numba.core.ir_utils import (get_call_table, mk_unique_var,
from numba.core.errors import NumbaValueError
from numba.core.utils import OPERATORS_TO_BUILTINS
from numba.np import numpy_support
def _get_const_two_irs(ir1, ir2, var):
    """get constant in either of two IRs if available
    otherwise, throw GuardException
    """
    var_const = guard(find_const, ir1, var)
    if var_const is not None:
        return var_const
    var_const = guard(find_const, ir2, var)
    if var_const is not None:
        return var_const
    raise GuardException