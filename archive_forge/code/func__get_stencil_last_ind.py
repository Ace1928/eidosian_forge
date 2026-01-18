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
def _get_stencil_last_ind(self, dim_size, end_length, gen_nodes, scope, loc):
    last_ind = dim_size
    if end_length != 0:
        index_const = ir.Var(scope, mk_unique_var('stencil_const_var'), loc)
        self.typemap[index_const.name] = types.intp
        if isinstance(end_length, numbers.Number):
            const_assign = ir.Assign(ir.Const(end_length, loc), index_const, loc)
        else:
            const_assign = ir.Assign(end_length, index_const, loc)
        gen_nodes.append(const_assign)
        last_ind = ir.Var(scope, mk_unique_var('last_ind'), loc)
        self.typemap[last_ind.name] = types.intp
        g_var = ir.Var(scope, mk_unique_var('compute_last_ind_var'), loc)
        check_func = numba.njit(_compute_last_ind)
        func_typ = types.functions.Dispatcher(check_func)
        self.typemap[g_var.name] = func_typ
        g_obj = ir.Global('_compute_last_ind', check_func, loc)
        g_assign = ir.Assign(g_obj, g_var, loc)
        gen_nodes.append(g_assign)
        index_call = ir.Expr.call(g_var, [dim_size, index_const], (), loc)
        self.calltypes[index_call] = func_typ.get_call_type(self.typingctx, [types.intp, types.intp], {})
        index_assign = ir.Assign(index_call, last_ind, loc)
        gen_nodes.append(index_assign)
    return last_ind