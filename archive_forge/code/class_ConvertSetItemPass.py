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
class ConvertSetItemPass:
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
                if isinstance(instr, (ir.StaticSetItem, ir.SetItem)):
                    loc = instr.loc
                    target = instr.target
                    index = get_index_var(instr)
                    value = instr.value
                    target_typ = pass_states.typemap[target.name]
                    index_typ = pass_states.typemap[index.name]
                    value_typ = pass_states.typemap[value.name]
                    if isinstance(target_typ, types.npytypes.Array):
                        if isinstance(index_typ, types.npytypes.Array) and isinstance(index_typ.dtype, types.Boolean) and (target_typ.ndim == index_typ.ndim):
                            if isinstance(value_typ, types.Number):
                                new_instr = self._setitem_to_parfor(equiv_set, loc, target, index, value)
                                self.rewritten.append(dict(old=instr, new=new_instr, reason='masked_assign_broadcast_scalar'))
                                instr = new_instr
                            elif isinstance(value_typ, types.npytypes.Array):
                                val_def = guard(get_definition, pass_states.func_ir, value.name)
                                if isinstance(val_def, ir.Expr) and val_def.op == 'getitem' and (val_def.index.name == index.name):
                                    new_instr = self._setitem_to_parfor(equiv_set, loc, target, index, val_def.value)
                                    self.rewritten.append(dict(old=instr, new=new_instr, reason='masked_assign_array'))
                                    instr = new_instr
                        else:
                            shape = equiv_set.get_shape(instr)
                            if isinstance(index_typ, types.BaseTuple):
                                sliced_dims = len(list(filter(lambda x: isinstance(x, types.misc.SliceType), index_typ.types)))
                            elif isinstance(index_typ, types.misc.SliceType):
                                sliced_dims = 1
                            else:
                                sliced_dims = 0
                            if shape is not None and (not isinstance(value_typ, types.npytypes.Array) or sliced_dims == value_typ.ndim):
                                new_instr = self._setitem_to_parfor(equiv_set, loc, target, index, value, shape=shape)
                                self.rewritten.append(dict(old=instr, new=new_instr, reason='slice'))
                                instr = new_instr
                new_body.append(instr)
            block.body = new_body

    def _setitem_to_parfor(self, equiv_set, loc, target, index, value, shape=None):
        """generate parfor from setitem node with a boolean or slice array indices.
        The value can be either a scalar or an array variable, and if a boolean index
        is used for the latter case, the same index must be used for the value too.
        """
        pass_states = self.pass_states
        scope = target.scope
        arr_typ = pass_states.typemap[target.name]
        el_typ = arr_typ.dtype
        index_typ = pass_states.typemap[index.name]
        init_block = ir.Block(scope, loc)
        if shape:
            assert isinstance(index_typ, types.BaseTuple) or isinstance(index_typ, types.SliceType)
            size_vars = shape
            subarr_var = ir.Var(scope, mk_unique_var('$subarr'), loc)
            getitem_call = ir.Expr.getitem(target, index, loc)
            subarr_typ = typing.arraydecl.get_array_index_type(arr_typ, index_typ).result
            pass_states.typemap[subarr_var.name] = subarr_typ
            pass_states.calltypes[getitem_call] = self._type_getitem((arr_typ, index_typ))
            init_block.append(ir.Assign(getitem_call, subarr_var, loc))
            target = subarr_var
        else:
            assert isinstance(index_typ, types.ArrayCompatible)
            size_vars = equiv_set.get_shape(target)
            bool_typ = index_typ.dtype
        loopnests = []
        index_vars = []
        for size_var in size_vars:
            index_var = ir.Var(scope, mk_unique_var('parfor_index'), loc)
            index_vars.append(index_var)
            pass_states.typemap[index_var.name] = types.uintp
            loopnests.append(LoopNest(index_var, 0, size_var, 1))
        body_label = next_label()
        body_block = ir.Block(scope, loc)
        index_var, index_var_typ = _make_index_var(pass_states.typemap, scope, index_vars, body_block)
        parfor = Parfor(loopnests, init_block, {}, loc, index_var, equiv_set, ('setitem', ''), pass_states.flags)
        if shape:
            parfor.loop_body = {body_label: body_block}
            true_block = body_block
            end_label = None
        else:
            true_label = next_label()
            true_block = ir.Block(scope, loc)
            end_label = next_label()
            end_block = ir.Block(scope, loc)
            parfor.loop_body = {body_label: body_block, true_label: true_block, end_label: end_block}
            mask_var = ir.Var(scope, mk_unique_var('$mask_var'), loc)
            pass_states.typemap[mask_var.name] = bool_typ
            mask_val = ir.Expr.getitem(index, index_var, loc)
            body_block.body.extend([ir.Assign(mask_val, mask_var, loc), ir.Branch(mask_var, true_label, end_label, loc)])
        value_typ = pass_states.typemap[value.name]
        if isinstance(value_typ, types.npytypes.Array):
            value_var = ir.Var(scope, mk_unique_var('$value_var'), loc)
            pass_states.typemap[value_var.name] = value_typ.dtype
            getitem_call = ir.Expr.getitem(value, index_var, loc)
            pass_states.calltypes[getitem_call] = signature(value_typ.dtype, value_typ, index_var_typ)
            true_block.body.append(ir.Assign(getitem_call, value_var, loc))
        else:
            value_var = value
        setitem_node = ir.SetItem(target, index_var, value_var, loc)
        pass_states.calltypes[setitem_node] = signature(types.none, pass_states.typemap[target.name], index_var_typ, el_typ)
        true_block.body.append(setitem_node)
        if end_label:
            true_block.body.append(ir.Jump(end_label, loc))
        if config.DEBUG_ARRAY_OPT >= 1:
            print('parfor from setitem')
            parfor.dump()
        return parfor

    def _type_getitem(self, args):
        fnty = operator.getitem
        return self.pass_states.typingctx.resolve_function_type(fnty, tuple(args), {})