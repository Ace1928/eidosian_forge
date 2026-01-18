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
class ConvertReducePass:
    """
    Find reduce() calls and convert them to parfors.
    """

    def __init__(self, pass_states):
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
                parfor = None
                if isinstance(instr, ir.Assign):
                    loc = instr.loc
                    lhs = instr.target
                    expr = instr.value
                    callname = guard(find_callname, pass_states.func_ir, expr)
                    if callname == ('reduce', 'builtins') or callname == ('reduce', '_functools'):
                        parfor = guard(self._reduce_to_parfor, equiv_set, lhs, expr.args, loc)
                    if parfor:
                        self.rewritten.append(dict(new=parfor, old=instr, reason='reduce'))
                        instr = parfor
                new_body.append(instr)
            block.body = new_body
        return

    def _reduce_to_parfor(self, equiv_set, lhs, args, loc):
        """
        Convert a reduce call to a parfor.
        The call arguments should be (call_name, array, init_value).
        """
        pass_states = self.pass_states
        scope = lhs.scope
        call_name = args[0]
        in_arr = args[1]
        arr_def = get_definition(pass_states.func_ir, in_arr.name)
        mask_var = None
        mask_indices = None
        mask_query_result = guard(_find_mask, pass_states.typemap, pass_states.func_ir, arr_def)
        if mask_query_result:
            in_arr, mask_var, mask_typ, mask_indices = mask_query_result
        init_val = args[2]
        size_vars = equiv_set.get_shape(in_arr if mask_indices is None else mask_var)
        if size_vars is None:
            return None
        index_vars, loopnests = _mk_parfor_loops(pass_states.typemap, size_vars, scope, loc)
        mask_index = index_vars
        if mask_indices:
            raise AssertionError('unreachable')
            index_vars = tuple((x if x else index_vars[0] for x in mask_indices))
        acc_var = lhs
        init_block = ir.Block(scope, loc)
        init_block.body.append(ir.Assign(init_val, acc_var, loc))
        body_label = next_label()
        index_var, loop_body = self._mk_reduction_body(call_name, scope, loc, index_vars, in_arr, acc_var)
        if mask_indices:
            raise AssertionError('unreachable')
            index_var = mask_index[0]
        if mask_var is not None:
            true_label = min(loop_body.keys())
            false_label = max(loop_body.keys())
            body_block = ir.Block(scope, loc)
            loop_body[body_label] = body_block
            mask = ir.Var(scope, mk_unique_var('$mask_val'), loc)
            pass_states.typemap[mask.name] = mask_typ
            mask_val = ir.Expr.getitem(mask_var, index_var, loc)
            body_block.body.extend([ir.Assign(mask_val, mask, loc), ir.Branch(mask, true_label, false_label, loc)])
        parfor = Parfor(loopnests, init_block, loop_body, loc, index_var, equiv_set, ('{} function'.format(call_name), 'reduction'), pass_states.flags)
        if config.DEBUG_ARRAY_OPT >= 1:
            print('parfor from reduction')
            parfor.dump()
        return parfor

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