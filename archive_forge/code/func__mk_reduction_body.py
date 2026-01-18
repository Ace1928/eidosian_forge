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
def _mk_reduction_body(self, call_name, scope, loc, index_vars, in_arr, acc_var):
    """
        Produce the body blocks for a reduction function indicated by call_name.
        """
    from numba.core.inline_closurecall import check_reduce_func
    pass_states = self.pass_states
    reduce_func = get_definition(pass_states.func_ir, call_name)
    fcode = check_reduce_func(pass_states.func_ir, reduce_func)
    arr_typ = pass_states.typemap[in_arr.name]
    in_typ = arr_typ.dtype
    body_block = ir.Block(scope, loc)
    index_var, index_var_type = _make_index_var(pass_states.typemap, scope, index_vars, body_block)
    tmp_var = ir.Var(scope, mk_unique_var('$val'), loc)
    pass_states.typemap[tmp_var.name] = in_typ
    getitem_call = ir.Expr.getitem(in_arr, index_var, loc)
    pass_states.calltypes[getitem_call] = signature(in_typ, arr_typ, index_var_type)
    body_block.append(ir.Assign(getitem_call, tmp_var, loc))
    reduce_f_ir = compile_to_numba_ir(fcode, pass_states.func_ir.func_id.func.__globals__, pass_states.typingctx, pass_states.targetctx, (in_typ, in_typ), pass_states.typemap, pass_states.calltypes)
    loop_body = reduce_f_ir.blocks
    end_label = next_label()
    end_block = ir.Block(scope, loc)
    loop_body[end_label] = end_block
    first_reduce_label = min(reduce_f_ir.blocks.keys())
    first_reduce_block = reduce_f_ir.blocks[first_reduce_label]
    body_block.body.extend(first_reduce_block.body)
    first_reduce_block.body = body_block.body
    replace_arg_nodes(first_reduce_block, [acc_var, tmp_var])
    replace_returns(loop_body, acc_var, end_label)
    return (index_var, loop_body)