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
class ConvertNumpyPass:
    """
    Convert supported Numpy functions, as well as arrayexpr nodes, to
    parfor nodes.
    """

    def __init__(self, pass_states):
        self.pass_states = pass_states
        self.rewritten = []

    def run(self, blocks):
        pass_states = self.pass_states
        topo_order = find_topo_order(blocks)
        avail_vars = []
        for label in topo_order:
            block = blocks[label]
            new_body = []
            equiv_set = pass_states.array_analysis.get_equiv_set(label)
            for instr in block.body:
                if isinstance(instr, ir.Assign):
                    expr = instr.value
                    lhs = instr.target
                    lhs_typ = self.pass_states.typemap[lhs.name]
                    if self._is_C_or_F_order(lhs_typ):
                        if guard(self._is_supported_npycall, expr):
                            new_instr = self._numpy_to_parfor(equiv_set, lhs, expr)
                            if new_instr is not None:
                                self.rewritten.append(dict(old=instr, new=new_instr, reason='numpy_allocator'))
                                instr = new_instr
                        elif isinstance(expr, ir.Expr) and expr.op == 'arrayexpr':
                            new_instr = self._arrayexpr_to_parfor(equiv_set, lhs, expr, avail_vars)
                            self.rewritten.append(dict(old=instr, new=new_instr, reason='arrayexpr'))
                            instr = new_instr
                    avail_vars.append(lhs.name)
                new_body.append(instr)
            block.body = new_body

    def _is_C_order(self, arr_name):
        if isinstance(arr_name, types.npytypes.Array):
            return arr_name.layout == 'C' and arr_name.ndim > 0
        elif arr_name is str:
            typ = self.pass_states.typemap[arr_name]
            return isinstance(typ, types.npytypes.Array) and typ.layout == 'C' and (typ.ndim > 0)
        else:
            return False

    def _is_C_or_F_order(self, arr_name):
        if isinstance(arr_name, types.npytypes.Array):
            return (arr_name.layout == 'C' or arr_name.layout == 'F') and arr_name.ndim > 0
        elif arr_name is str:
            typ = self.pass_states.typemap[arr_name]
            return isinstance(typ, types.npytypes.Array) and (typ.layout == 'C' or typ.layout == 'F') and (typ.ndim > 0)
        else:
            return False

    def _arrayexpr_to_parfor(self, equiv_set, lhs, arrayexpr, avail_vars):
        """generate parfor from arrayexpr node, which is essentially a
        map with recursive tree.
        """
        pass_states = self.pass_states
        scope = lhs.scope
        loc = lhs.loc
        expr = arrayexpr.expr
        arr_typ = pass_states.typemap[lhs.name]
        el_typ = arr_typ.dtype
        size_vars = equiv_set.get_shape(lhs)
        index_vars, loopnests = _mk_parfor_loops(pass_states.typemap, size_vars, scope, loc)
        init_block = ir.Block(scope, loc)
        init_block.body = mk_alloc(pass_states.typingctx, pass_states.typemap, pass_states.calltypes, lhs, tuple(size_vars), el_typ, scope, loc, pass_states.typemap[lhs.name])
        body_label = next_label()
        body_block = ir.Block(scope, loc)
        expr_out_var = ir.Var(scope, mk_unique_var('$expr_out_var'), loc)
        pass_states.typemap[expr_out_var.name] = el_typ
        index_var, index_var_typ = _make_index_var(pass_states.typemap, scope, index_vars, body_block)
        body_block.body.extend(_arrayexpr_tree_to_ir(pass_states.func_ir, pass_states.typingctx, pass_states.typemap, pass_states.calltypes, equiv_set, init_block, expr_out_var, expr, index_var, index_vars, avail_vars))
        pat = ('array expression {}'.format(repr_arrayexpr(arrayexpr.expr)),)
        parfor = Parfor(loopnests, init_block, {}, loc, index_var, equiv_set, pat[0], pass_states.flags)
        setitem_node = ir.SetItem(lhs, index_var, expr_out_var, loc)
        pass_states.calltypes[setitem_node] = signature(types.none, pass_states.typemap[lhs.name], index_var_typ, el_typ)
        body_block.body.append(setitem_node)
        parfor.loop_body = {body_label: body_block}
        if config.DEBUG_ARRAY_OPT >= 1:
            print('parfor from arrayexpr')
            parfor.dump()
        return parfor

    def _is_supported_npycall(self, expr):
        """check if we support parfor translation for
        this Numpy call.
        """
        call_name, mod_name = find_callname(self.pass_states.func_ir, expr)
        if not (isinstance(mod_name, str) and mod_name.startswith('numpy')):
            return False
        if call_name in ['zeros', 'ones']:
            return True
        if mod_name == 'numpy.random' and call_name in random_calls:
            return True
        return False

    def _numpy_to_parfor(self, equiv_set, lhs, expr):
        call_name, mod_name = find_callname(self.pass_states.func_ir, expr)
        args = expr.args
        kws = dict(expr.kws)
        if call_name in ['zeros', 'ones'] or mod_name == 'numpy.random':
            return self._numpy_map_to_parfor(equiv_set, call_name, lhs, args, kws, expr)
        raise errors.UnsupportedRewriteError(f'parfor translation failed for {expr}', loc=expr.loc)

    def _numpy_map_to_parfor(self, equiv_set, call_name, lhs, args, kws, expr):
        """generate parfor from Numpy calls that are maps.
        """
        pass_states = self.pass_states
        scope = lhs.scope
        loc = lhs.loc
        arr_typ = pass_states.typemap[lhs.name]
        el_typ = arr_typ.dtype
        size_vars = equiv_set.get_shape(lhs)
        if size_vars is None:
            if config.DEBUG_ARRAY_OPT >= 1:
                print('Could not convert numpy map to parfor, unknown size')
            return None
        index_vars, loopnests = _mk_parfor_loops(pass_states.typemap, size_vars, scope, loc)
        init_block = ir.Block(scope, loc)
        init_block.body = mk_alloc(pass_states.typingctx, pass_states.typemap, pass_states.calltypes, lhs, tuple(size_vars), el_typ, scope, loc, pass_states.typemap[lhs.name])
        body_label = next_label()
        body_block = ir.Block(scope, loc)
        expr_out_var = ir.Var(scope, mk_unique_var('$expr_out_var'), loc)
        pass_states.typemap[expr_out_var.name] = el_typ
        index_var, index_var_typ = _make_index_var(pass_states.typemap, scope, index_vars, body_block)
        if call_name == 'zeros':
            value = ir.Const(el_typ(0), loc)
        elif call_name == 'ones':
            value = ir.Const(el_typ(1), loc)
        elif call_name in random_calls:
            _remove_size_arg(call_name, expr)
            new_arg_typs, new_kw_types = _get_call_arg_types(expr, pass_states.typemap)
            pass_states.calltypes.pop(expr)
            pass_states.calltypes[expr] = pass_states.typemap[expr.func.name].get_call_type(typing.Context(), new_arg_typs, new_kw_types)
            value = expr
        else:
            raise NotImplementedError('Map of numpy.{} to parfor is not implemented'.format(call_name))
        value_assign = ir.Assign(value, expr_out_var, loc)
        body_block.body.append(value_assign)
        setitem_node = ir.SetItem(lhs, index_var, expr_out_var, loc)
        pass_states.calltypes[setitem_node] = signature(types.none, pass_states.typemap[lhs.name], index_var_typ, el_typ)
        body_block.body.append(setitem_node)
        parfor = Parfor(loopnests, init_block, {}, loc, index_var, equiv_set, ('{} function'.format(call_name), 'NumPy mapping'), pass_states.flags)
        parfor.loop_body = {body_label: body_block}
        if config.DEBUG_ARRAY_OPT >= 1:
            print('generated parfor for numpy map:')
            parfor.dump()
        return parfor