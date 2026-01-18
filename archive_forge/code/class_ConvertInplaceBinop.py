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
class ConvertInplaceBinop:
    """Parfor subpass to convert setitem on Arrays
    """

    def __init__(self, pass_states):
        """
        Parameters
        ----------
        pass_states : ParforPassStates
        """
        self.pass_states = pass_states
        self.rewritten = []

    def run(self, blocks):
        pass_states = self.pass_states
        topo_order = find_topo_order(blocks)
        for label in topo_order:
            block = blocks[label]
            new_body = []
            equiv_set = pass_states.array_analysis.get_equiv_set(label)
            for instr in block.body:
                if isinstance(instr, ir.Assign):
                    lhs = instr.target
                    expr = instr.value
                    if isinstance(expr, ir.Expr) and expr.op == 'inplace_binop':
                        loc = expr.loc
                        target = expr.lhs
                        value = expr.rhs
                        target_typ = pass_states.typemap[target.name]
                        value_typ = pass_states.typemap[value.name]
                        if isinstance(target_typ, types.npytypes.Array):
                            if isinstance(value_typ, types.npytypes.Array):
                                new_instr = self._inplace_binop_to_parfor(equiv_set, loc, expr.immutable_fn, target, value)
                                self.rewritten.append(dict(old=instr, new=new_instr, reason='inplace_binop'))
                                instr = [new_instr, ir.Assign(target, lhs, loc)]
                if isinstance(instr, list):
                    new_body.extend(instr)
                else:
                    new_body.append(instr)
            block.body = new_body

    def _inplace_binop_to_parfor(self, equiv_set, loc, op, target, value):
        """generate parfor from setitem node with a boolean or slice array indices.
        The value can be either a scalar or an array variable, and if a boolean index
        is used for the latter case, the same index must be used for the value too.
        """
        pass_states = self.pass_states
        scope = target.scope
        arr_typ = pass_states.typemap[target.name]
        el_typ = arr_typ.dtype
        init_block = ir.Block(scope, loc)
        value_typ = pass_states.typemap[value.name]
        size_vars = equiv_set.get_shape(target)
        index_vars, loopnests = _mk_parfor_loops(pass_states.typemap, size_vars, scope, loc)
        body_label = next_label()
        body_block = ir.Block(scope, loc)
        index_var, index_var_typ = _make_index_var(pass_states.typemap, scope, index_vars, body_block)
        value_var = ir.Var(scope, mk_unique_var('$value_var'), loc)
        pass_states.typemap[value_var.name] = value_typ.dtype
        getitem_call = ir.Expr.getitem(value, index_var, loc)
        pass_states.calltypes[getitem_call] = signature(value_typ.dtype, value_typ, index_var_typ)
        body_block.body.append(ir.Assign(getitem_call, value_var, loc))
        target_var = ir.Var(scope, mk_unique_var('$target_var'), loc)
        pass_states.typemap[target_var.name] = el_typ
        getitem_call = ir.Expr.getitem(target, index_var, loc)
        pass_states.calltypes[getitem_call] = signature(el_typ, arr_typ, index_var_typ)
        body_block.body.append(ir.Assign(getitem_call, target_var, loc))
        expr_out_var = ir.Var(scope, mk_unique_var('$expr_out_var'), loc)
        pass_states.typemap[expr_out_var.name] = el_typ
        binop_expr = ir.Expr.binop(op, target_var, value_var, loc)
        body_block.body.append(ir.Assign(binop_expr, expr_out_var, loc))
        unified_type = self.pass_states.typingctx.unify_pairs(el_typ, value_typ.dtype)
        pass_states.calltypes[binop_expr] = signature(unified_type, unified_type, unified_type)
        setitem_node = ir.SetItem(target, index_var, expr_out_var, loc)
        pass_states.calltypes[setitem_node] = signature(types.none, arr_typ, index_var_typ, el_typ)
        body_block.body.append(setitem_node)
        parfor = Parfor(loopnests, init_block, {}, loc, index_var, equiv_set, ('inplace_binop', ''), pass_states.flags)
        parfor.loop_body = {body_label: body_block}
        if config.DEBUG_ARRAY_OPT >= 1:
            print('parfor from inplace_binop')
            parfor.dump()
        return parfor

    def _type_getitem(self, args):
        fnty = operator.getitem
        return self.pass_states.typingctx.resolve_function_type(fnty, tuple(args), {})