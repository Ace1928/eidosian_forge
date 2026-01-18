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
def _get_const_binary_expr(stencil_ir, func_ir, index_def):
    """evaluate constant binary expr if possible
    otherwise, raise GuardException
    """
    require(isinstance(index_def, ir.Expr) and index_def.op == 'binop')
    arg1 = _get_const_index_expr_inner(stencil_ir, func_ir, index_def.lhs)
    arg2 = _get_const_index_expr_inner(stencil_ir, func_ir, index_def.rhs)
    op = OPERATORS_TO_BUILTINS[index_def.fn]
    return eval('{}{}{}'.format(arg1, op, arg2))