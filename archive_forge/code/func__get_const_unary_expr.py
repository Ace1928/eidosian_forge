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
def _get_const_unary_expr(stencil_ir, func_ir, index_def):
    """evaluate constant unary expr if possible
    otherwise, raise GuardException
    """
    require(isinstance(index_def, ir.Expr) and index_def.op == 'unary')
    inner_var = index_def.value
    const_val = _get_const_index_expr_inner(stencil_ir, func_ir, inner_var)
    op = OPERATORS_TO_BUILTINS[index_def.fn]
    return eval('{}{}'.format(op, const_val))