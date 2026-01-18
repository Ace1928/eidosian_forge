import types as pytypes  # avoid confusion with numba.types
import sys, math
import os
import textwrap
import copy
import inspect
import linecache
from functools import reduce
from collections import defaultdict, OrderedDict, namedtuple
from contextlib import contextmanager
import operator
from dataclasses import make_dataclass
import warnings
from llvmlite import ir as lir
from numba.core.imputils import impl_ret_untracked
import numba.core.ir
from numba.core import types, typing, utils, errors, ir, analysis, postproc, rewrites, typeinfer, config, ir_utils
from numba import prange, pndindex
from numba.np.npdatetime_helpers import datetime_minimum, datetime_maximum
from numba.np.numpy_support import as_dtype, numpy_version
from numba.core.typing.templates import infer_global, AbstractTemplate
from numba.stencils.stencilparfor import StencilPass
from numba.core.extending import register_jitable, lower_builtin
from numba.core.ir_utils import (
from numba.core.analysis import (compute_use_defs, compute_live_map,
from numba.core.controlflow import CFGraph
from numba.core.typing import npydecl, signature
from numba.core.types.functions import Function
from numba.parfors.array_analysis import (random_int_args, random_1arg_size,
from numba.core.extending import overload
import copy
import numpy
import numpy as np
from numba.parfors import array_analysis
import numba.cpython.builtins
from numba.stencils import stencilparfor
def _gen_np_divide(arg1, arg2, out_ir, typemap):
    """generate np.divide() instead of / for array_expr to get numpy error model
    like inf for division by zero (test_division_by_zero).
    """
    scope = arg1.scope
    loc = arg1.loc
    g_np_var = ir.Var(scope, mk_unique_var('$np_g_var'), loc)
    typemap[g_np_var.name] = types.misc.Module(numpy)
    g_np = ir.Global('np', numpy, loc)
    g_np_assign = ir.Assign(g_np, g_np_var, loc)
    div_attr_call = ir.Expr.getattr(g_np_var, 'divide', loc)
    attr_var = ir.Var(scope, mk_unique_var('$div_attr'), loc)
    func_var_typ = get_np_ufunc_typ(numpy.divide)
    typemap[attr_var.name] = func_var_typ
    attr_assign = ir.Assign(div_attr_call, attr_var, loc)
    div_call = ir.Expr.call(attr_var, [arg1, arg2], (), loc)
    func_typ = func_var_typ.get_call_type(typing.Context(), [typemap[arg1.name], typemap[arg2.name]], {})
    out_ir.extend([g_np_assign, attr_assign])
    return (func_typ, div_call)