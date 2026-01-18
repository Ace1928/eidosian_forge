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
def _get_const_index_expr_inner(stencil_ir, func_ir, index_var):
    """inner constant inference function that calls constant, unary and binary
    cases.
    """
    require(isinstance(index_var, ir.Var))
    var_const = guard(_get_const_two_irs, stencil_ir, func_ir, index_var)
    if var_const is not None:
        return var_const
    index_def = ir_utils.get_definition(stencil_ir, index_var)
    var_const = guard(_get_const_unary_expr, stencil_ir, func_ir, index_def)
    if var_const is not None:
        return var_const
    var_const = guard(_get_const_binary_expr, stencil_ir, func_ir, index_def)
    if var_const is not None:
        return var_const
    raise GuardException