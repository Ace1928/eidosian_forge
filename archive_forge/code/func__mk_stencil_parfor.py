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
def _mk_stencil_parfor(self, label, in_args, out_arr, stencil_ir, index_offsets, target, return_type, stencil_func, arg_to_arr_dict):
    """ Converts a set of stencil kernel blocks to a parfor.
        """
    gen_nodes = []
    stencil_blocks = stencil_ir.blocks
    if config.DEBUG_ARRAY_OPT >= 1:
        print('_mk_stencil_parfor', label, in_args, out_arr, index_offsets, return_type, stencil_func, stencil_blocks)
        ir_utils.dump_blocks(stencil_blocks)
    in_arr = in_args[0]
    in_arr_typ = self.typemap[in_arr.name]
    in_cps, out_cps = ir_utils.copy_propagate(stencil_blocks, self.typemap)
    name_var_table = ir_utils.get_name_var_table(stencil_blocks)
    ir_utils.apply_copy_propagate(stencil_blocks, in_cps, name_var_table, self.typemap, self.calltypes)
    if config.DEBUG_ARRAY_OPT >= 1:
        print('stencil_blocks after copy_propagate')
        ir_utils.dump_blocks(stencil_blocks)
    ir_utils.remove_dead(stencil_blocks, self.func_ir.arg_names, stencil_ir, self.typemap)
    if config.DEBUG_ARRAY_OPT >= 1:
        print('stencil_blocks after removing dead code')
        ir_utils.dump_blocks(stencil_blocks)
    ndims = self.typemap[in_arr.name].ndim
    scope = in_arr.scope
    loc = in_arr.loc
    parfor_vars = []
    for i in range(ndims):
        parfor_var = ir.Var(scope, mk_unique_var('$parfor_index_var'), loc)
        self.typemap[parfor_var.name] = types.intp
        parfor_vars.append(parfor_var)
    start_lengths, end_lengths = self._replace_stencil_accesses(stencil_ir, parfor_vars, in_args, index_offsets, stencil_func, arg_to_arr_dict)
    if config.DEBUG_ARRAY_OPT >= 1:
        print('stencil_blocks after replace stencil accesses')
        print('start_lengths:', start_lengths)
        print('end_lengths:', end_lengths)
        ir_utils.dump_blocks(stencil_blocks)
    loopnests = []
    equiv_set = self.array_analysis.get_equiv_set(label)
    in_arr_dim_sizes = equiv_set.get_shape(in_arr)
    assert ndims == len(in_arr_dim_sizes)
    start_inds = []
    last_inds = []
    for i in range(ndims):
        last_ind = self._get_stencil_last_ind(in_arr_dim_sizes[i], end_lengths[i], gen_nodes, scope, loc)
        start_ind = self._get_stencil_start_ind(start_lengths[i], gen_nodes, scope, loc)
        start_inds.append(start_ind)
        last_inds.append(last_ind)
        loopnests.append(numba.parfors.parfor.LoopNest(parfor_vars[i], start_ind, last_ind, 1))
    parfor_body_exit_label = max(stencil_blocks.keys()) + 1
    stencil_blocks[parfor_body_exit_label] = ir.Block(scope, loc)
    exit_value_var = ir.Var(scope, mk_unique_var('$parfor_exit_value'), loc)
    self.typemap[exit_value_var.name] = return_type.dtype
    for_replacing_ret = []
    if ndims == 1:
        parfor_ind_var = parfor_vars[0]
    else:
        parfor_ind_var = ir.Var(scope, mk_unique_var('$parfor_index_tuple_var'), loc)
        self.typemap[parfor_ind_var.name] = types.containers.UniTuple(types.intp, ndims)
        tuple_call = ir.Expr.build_tuple(parfor_vars, loc)
        tuple_assign = ir.Assign(tuple_call, parfor_ind_var, loc)
        for_replacing_ret.append(tuple_assign)
    if config.DEBUG_ARRAY_OPT >= 1:
        print('stencil_blocks after creating parfor index var')
        ir_utils.dump_blocks(stencil_blocks)
    init_block = ir.Block(scope, loc)
    if out_arr is None:
        in_arr_typ = self.typemap[in_arr.name]
        shape_name = ir_utils.mk_unique_var('in_arr_shape')
        shape_var = ir.Var(scope, shape_name, loc)
        shape_getattr = ir.Expr.getattr(in_arr, 'shape', loc)
        self.typemap[shape_name] = types.containers.UniTuple(types.intp, in_arr_typ.ndim)
        init_block.body.extend([ir.Assign(shape_getattr, shape_var, loc)])
        zero_name = ir_utils.mk_unique_var('zero_val')
        zero_var = ir.Var(scope, zero_name, loc)
        if 'cval' in stencil_func.options:
            cval = stencil_func.options['cval']
            cval_ty = typing.typeof.typeof(cval)
            if not self.typingctx.can_convert(cval_ty, return_type.dtype):
                raise ValueError('cval type does not match stencil return type.')
            temp2 = return_type.dtype(cval)
        else:
            temp2 = return_type.dtype(0)
        full_const = ir.Const(temp2, loc)
        self.typemap[zero_name] = return_type.dtype
        init_block.body.extend([ir.Assign(full_const, zero_var, loc)])
        so_name = ir_utils.mk_unique_var('stencil_output')
        out_arr = ir.Var(scope, so_name, loc)
        self.typemap[out_arr.name] = numba.core.types.npytypes.Array(return_type.dtype, in_arr_typ.ndim, in_arr_typ.layout)
        dtype_g_np_var = ir.Var(scope, mk_unique_var('$np_g_var'), loc)
        self.typemap[dtype_g_np_var.name] = types.misc.Module(np)
        dtype_g_np = ir.Global('np', np, loc)
        dtype_g_np_assign = ir.Assign(dtype_g_np, dtype_g_np_var, loc)
        init_block.body.append(dtype_g_np_assign)
        return_type_name = numpy_support.as_dtype(return_type.dtype).type.__name__
        if return_type_name == 'bool':
            return_type_name = 'bool_'
        dtype_np_attr_call = ir.Expr.getattr(dtype_g_np_var, return_type_name, loc)
        dtype_attr_var = ir.Var(scope, mk_unique_var('$np_attr_attr'), loc)
        self.typemap[dtype_attr_var.name] = types.functions.NumberClass(return_type.dtype)
        dtype_attr_assign = ir.Assign(dtype_np_attr_call, dtype_attr_var, loc)
        init_block.body.append(dtype_attr_assign)
        stmts = ir_utils.gen_np_call('empty', np.empty, out_arr, [shape_var, dtype_attr_var], self.typingctx, self.typemap, self.calltypes)
        none_var = ir.Var(scope, mk_unique_var('$none_var'), loc)
        none_assign = ir.Assign(ir.Const(None, loc), none_var, loc)
        stmts.append(none_assign)
        self.typemap[none_var.name] = types.none
        zero_index_var = ir.Var(scope, mk_unique_var('$zero_index_var'), loc)
        zero_index_assign = ir.Assign(ir.Const(0, loc), zero_index_var, loc)
        stmts.append(zero_index_assign)
        self.typemap[zero_index_var.name] = types.intp
        slice_func_var = ir.Var(scope, mk_unique_var('$slice_func_var'), loc)
        slice_fn_ty = self.typingctx.resolve_value_type(slice)
        self.typemap[slice_func_var.name] = slice_fn_ty
        slice_g = ir.Global('slice', slice, loc)
        slice_assign = ir.Assign(slice_g, slice_func_var, loc)
        stmts.append(slice_assign)
        sig = self.typingctx.resolve_function_type(slice_fn_ty, (types.none,) * 2, {})
        slice_callexpr = ir.Expr.call(func=slice_func_var, args=(none_var, none_var), kws=(), loc=loc)
        self.calltypes[slice_callexpr] = sig
        slice_var = ir.Var(scope, mk_unique_var('$slice'), loc)
        self.typemap[slice_var.name] = types.slice2_type
        slice_assign = ir.Assign(slice_callexpr, slice_var, loc)
        stmts.append(slice_assign)

        def handle_border(slice_fn_ty, dim, scope, loc, slice_func_var, stmts, border_inds, border_tuple_items, other_arg, other_first):
            sig = self.typingctx.resolve_function_type(slice_fn_ty, (types.intp,) * 2, {})
            si = border_inds[dim]
            assert isinstance(si, (int, ir.Var))
            si_var = ir.Var(scope, mk_unique_var('$border_ind'), loc)
            self.typemap[si_var.name] = types.intp
            if isinstance(si, int):
                si_assign = ir.Assign(ir.Const(si, loc), si_var, loc)
            else:
                si_assign = ir.Assign(si, si_var, loc)
            stmts.append(si_assign)
            slice_callexpr = ir.Expr.call(func=slice_func_var, args=(other_arg, si_var) if other_first else (si_var, other_arg), kws=(), loc=loc)
            self.calltypes[slice_callexpr] = sig
            border_slice_var = ir.Var(scope, mk_unique_var('$slice'), loc)
            self.typemap[border_slice_var.name] = types.slice2_type
            slice_assign = ir.Assign(slice_callexpr, border_slice_var, loc)
            stmts.append(slice_assign)
            border_tuple_items[dim] = border_slice_var
            border_ind_var = ir.Var(scope, mk_unique_var('$border_index_tuple_var'), loc)
            self.typemap[border_ind_var.name] = types.containers.UniTuple(types.slice2_type, ndims)
            tuple_call = ir.Expr.build_tuple(border_tuple_items, loc)
            tuple_assign = ir.Assign(tuple_call, border_ind_var, loc)
            stmts.append(tuple_assign)
            setitem_call = ir.SetItem(out_arr, border_ind_var, zero_var, loc)
            self.calltypes[setitem_call] = signature(types.none, self.typemap[out_arr.name], self.typemap[border_ind_var.name], self.typemap[out_arr.name].dtype)
            stmts.append(setitem_call)
        for dim in range(in_arr_typ.ndim):
            start_tuple_items = [slice_var] * in_arr_typ.ndim
            last_tuple_items = [slice_var] * in_arr_typ.ndim
            handle_border(slice_fn_ty, dim, scope, loc, slice_func_var, stmts, start_inds, start_tuple_items, zero_index_var, True)
            handle_border(slice_fn_ty, dim, scope, loc, slice_func_var, stmts, last_inds, last_tuple_items, in_arr_dim_sizes[dim], False)
        equiv_set.insert_equiv(out_arr, in_arr_dim_sizes)
        init_block.body.extend(stmts)
    elif 'cval' in stencil_func.options:
        cval = stencil_func.options['cval']
        cval_ty = typing.typeof.typeof(cval)
        if not self.typingctx.can_convert(cval_ty, return_type.dtype):
            msg = 'cval type does not match stencil return type.'
            raise NumbaValueError(msg)
        slice_var = ir.Var(scope, mk_unique_var('$py_g_var'), loc)
        slice_fn_ty = self.typingctx.resolve_value_type(slice)
        self.typemap[slice_var.name] = slice_fn_ty
        slice_g = ir.Global('slice', slice, loc)
        slice_assigned = ir.Assign(slice_g, slice_var, loc)
        init_block.body.append(slice_assigned)
        sig = self.typingctx.resolve_function_type(slice_fn_ty, (types.none,) * 2, {})
        callexpr = ir.Expr.call(func=slice_var, args=(), kws=(), loc=loc)
        self.calltypes[callexpr] = sig
        slice_inst_var = ir.Var(scope, mk_unique_var('$slice_inst'), loc)
        self.typemap[slice_inst_var.name] = types.slice2_type
        slice_assign = ir.Assign(callexpr, slice_inst_var, loc)
        init_block.body.append(slice_assign)
        cval_const_val = ir.Const(return_type.dtype(cval), loc)
        cval_const_var = ir.Var(scope, mk_unique_var('$cval_const'), loc)
        self.typemap[cval_const_var.name] = return_type.dtype
        cval_const_assign = ir.Assign(cval_const_val, cval_const_var, loc)
        init_block.body.append(cval_const_assign)
        setitemexpr = ir.StaticSetItem(out_arr, slice(None, None), slice_inst_var, cval_const_var, loc)
        init_block.body.append(setitemexpr)
        sig = signature(types.none, self.typemap[out_arr.name], self.typemap[slice_inst_var.name], self.typemap[out_arr.name].dtype)
        self.calltypes[setitemexpr] = sig
    self.replace_return_with_setitem(stencil_blocks, exit_value_var, parfor_body_exit_label)
    if config.DEBUG_ARRAY_OPT >= 1:
        print('stencil_blocks after replacing return')
        ir_utils.dump_blocks(stencil_blocks)
    setitem_call = ir.SetItem(out_arr, parfor_ind_var, exit_value_var, loc)
    self.calltypes[setitem_call] = signature(types.none, self.typemap[out_arr.name], self.typemap[parfor_ind_var.name], self.typemap[out_arr.name].dtype)
    stencil_blocks[parfor_body_exit_label].body.extend(for_replacing_ret)
    stencil_blocks[parfor_body_exit_label].body.append(setitem_call)
    dummy_loc = ir.Loc('stencilparfor_dummy', -1)
    ret_const_var = ir.Var(scope, mk_unique_var('$cval_const'), dummy_loc)
    cval_const_assign = ir.Assign(ir.Const(0, loc=dummy_loc), ret_const_var, dummy_loc)
    stencil_blocks[parfor_body_exit_label].body.append(cval_const_assign)
    stencil_blocks[parfor_body_exit_label].body.append(ir.Return(ret_const_var, dummy_loc))
    stencil_blocks = ir_utils.simplify_CFG(stencil_blocks)
    stencil_blocks[max(stencil_blocks.keys())].body.pop()
    if config.DEBUG_ARRAY_OPT >= 1:
        print('stencil_blocks after adding SetItem')
        ir_utils.dump_blocks(stencil_blocks)
    pattern = ('stencil', [start_lengths, end_lengths])
    parfor = numba.parfors.parfor.Parfor(loopnests, init_block, stencil_blocks, loc, parfor_ind_var, equiv_set, pattern, self.flags)
    gen_nodes.append(parfor)
    gen_nodes.append(ir.Assign(out_arr, target, loc))
    return gen_nodes