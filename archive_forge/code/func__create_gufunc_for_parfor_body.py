import copy
import operator
import types as pytypes
import operator
import warnings
from dataclasses import make_dataclass
import llvmlite.ir
import numpy as np
import numba
from numba.parfors import parfor
from numba.core import types, ir, config, compiler, sigutils, cgutils
from numba.core.ir_utils import (
from numba.core.typing import signature
from numba.core import lowering
from numba.parfors.parfor import ensure_parallel_support
from numba.core.errors import (
from numba.parfors.parfor_lowering_utils import ParforLoweringBuilder
def _create_gufunc_for_parfor_body(lowerer, parfor, typemap, typingctx, targetctx, flags, locals, has_aliases, index_var_typ, races):
    """
    Takes a parfor and creates a gufunc function for its body.
    There are two parts to this function.
    1) Code to iterate across the iteration space as defined by the schedule.
    2) The parfor body that does the work for a single point in the iteration space.
    Part 1 is created as Python text for simplicity with a sentinel assignment to mark the point
    in the IR where the parfor body should be added.
    This Python text is 'exec'ed into existence and its IR retrieved with run_frontend.
    The IR is scanned for the sentinel assignment where that basic block is split and the IR
    for the parfor body inserted.
    """
    if config.DEBUG_ARRAY_OPT >= 1:
        print('starting _create_gufunc_for_parfor_body')
    loc = parfor.init_block.loc
    loop_body = copy.copy(parfor.loop_body)
    remove_dels(loop_body)
    parfor_dim = len(parfor.loop_nests)
    loop_indices = [l.index_variable.name for l in parfor.loop_nests]
    parfor_params = parfor.params
    parfor_outputs = numba.parfors.parfor.get_parfor_outputs(parfor, parfor_params)
    typemap = lowerer.fndesc.typemap
    parfor_redvars, parfor_reddict = numba.parfors.parfor.get_parfor_reductions(lowerer.func_ir, parfor, parfor_params, lowerer.fndesc.calltypes)
    parfor_inputs = sorted(list(set(parfor_params) - set(parfor_outputs) - set(parfor_redvars)))
    if config.DEBUG_ARRAY_OPT >= 1:
        print('parfor_params = ', parfor_params, ' ', type(parfor_params))
        print('parfor_outputs = ', parfor_outputs, ' ', type(parfor_outputs))
        print('parfor_inputs = ', parfor_inputs, ' ', type(parfor_inputs))
        print('parfor_redvars = ', parfor_redvars, ' ', type(parfor_redvars))
    tuple_expanded_parfor_inputs = []
    tuple_var_to_expanded_names = {}
    expanded_name_to_tuple_var = {}
    next_expanded_tuple_var = 0
    parfor_tuple_params = []
    for pi in parfor_inputs:
        pi_type = typemap[pi]
        if isinstance(pi_type, types.UniTuple) or isinstance(pi_type, types.NamedUniTuple):
            tuple_count = pi_type.count
            tuple_dtype = pi_type.dtype
            assert tuple_count <= config.PARFOR_MAX_TUPLE_SIZE
            this_var_expansion = []
            for i in range(tuple_count):
                expanded_name = 'expanded_tuple_var_' + str(next_expanded_tuple_var)
                tuple_expanded_parfor_inputs.append(expanded_name)
                this_var_expansion.append(expanded_name)
                expanded_name_to_tuple_var[expanded_name] = (pi, i)
                next_expanded_tuple_var += 1
                typemap[expanded_name] = tuple_dtype
            tuple_var_to_expanded_names[pi] = this_var_expansion
            parfor_tuple_params.append(pi)
        elif isinstance(pi_type, types.Tuple) or isinstance(pi_type, types.NamedTuple):
            tuple_count = pi_type.count
            tuple_types = pi_type.types
            assert tuple_count <= config.PARFOR_MAX_TUPLE_SIZE
            this_var_expansion = []
            for i in range(tuple_count):
                expanded_name = 'expanded_tuple_var_' + str(next_expanded_tuple_var)
                tuple_expanded_parfor_inputs.append(expanded_name)
                this_var_expansion.append(expanded_name)
                expanded_name_to_tuple_var[expanded_name] = (pi, i)
                next_expanded_tuple_var += 1
                typemap[expanded_name] = tuple_types[i]
            tuple_var_to_expanded_names[pi] = this_var_expansion
            parfor_tuple_params.append(pi)
        else:
            tuple_expanded_parfor_inputs.append(pi)
    parfor_inputs = tuple_expanded_parfor_inputs
    if config.DEBUG_ARRAY_OPT >= 1:
        print('parfor_inputs post tuple handling = ', parfor_inputs, ' ', type(parfor_inputs))
    races = races.difference(set(parfor_redvars))
    for race in races:
        msg = 'Variable %s used in parallel loop may be written to simultaneously by multiple workers and may result in non-deterministic or unintended results.' % race
        warnings.warn(NumbaParallelSafetyWarning(msg, loc))
    replace_var_with_array(races, loop_body, typemap, lowerer.fndesc.calltypes)
    parfor_redarrs = []
    parfor_red_arg_types = []
    for var in parfor_redvars:
        arr = var + '_arr'
        parfor_redarrs.append(arr)
        redarraytype = redtyp_to_redarraytype(typemap[var])
        parfor_red_arg_types.append(redarraytype)
        redarrsig = redarraytype_to_sig(redarraytype)
        if arr in typemap:
            assert typemap[arr] == redarrsig
        else:
            typemap[arr] = redarrsig
    parfor_params = parfor_inputs + parfor_outputs + parfor_redarrs
    if config.DEBUG_ARRAY_OPT >= 1:
        print('parfor_params = ', parfor_params, ' ', type(parfor_params))
        print('loop_indices = ', loop_indices, ' ', type(loop_indices))
        print('loop_body = ', loop_body, ' ', type(loop_body))
        _print_body(loop_body)
    param_dict = legalize_names_with_typemap(parfor_params + parfor_redvars + parfor_tuple_params, typemap)
    if config.DEBUG_ARRAY_OPT >= 1:
        print('param_dict = ', sorted(param_dict.items()), ' ', type(param_dict))
    ind_dict = legalize_names_with_typemap(loop_indices, typemap)
    legal_loop_indices = [ind_dict[v] for v in loop_indices]
    if config.DEBUG_ARRAY_OPT >= 1:
        print('ind_dict = ', sorted(ind_dict.items()), ' ', type(ind_dict))
        print('legal_loop_indices = ', legal_loop_indices, ' ', type(legal_loop_indices))
        for pd in parfor_params:
            print('pd = ', pd)
            print('pd type = ', typemap[pd], ' ', type(typemap[pd]))
    param_types = [to_scalar_from_0d(typemap[v]) for v in parfor_params]
    func_arg_types = [typemap[v] for v in parfor_inputs + parfor_outputs] + parfor_red_arg_types
    if config.DEBUG_ARRAY_OPT >= 1:
        print('new param_types:', param_types)
        print('new func_arg_types:', func_arg_types)
    replace_var_names(loop_body, param_dict)
    parfor_args = parfor_params
    parfor_params = [param_dict[v] for v in parfor_params]
    parfor_params_orig = parfor_params
    parfor_params = []
    ascontig = False
    for pindex in range(len(parfor_params_orig)):
        if ascontig and pindex < len(parfor_inputs) and isinstance(param_types[pindex], types.npytypes.Array):
            parfor_params.append(parfor_params_orig[pindex] + 'param')
        else:
            parfor_params.append(parfor_params_orig[pindex])
    replace_var_names(loop_body, ind_dict)
    loop_body_var_table = get_name_var_table(loop_body)
    sentinel_name = get_unused_var_name('__sentinel__', loop_body_var_table)
    if config.DEBUG_ARRAY_OPT >= 1:
        print('legal parfor_params = ', parfor_params, ' ', type(parfor_params))
    gufunc_name = '__numba_parfor_gufunc_%s' % hex(hash(parfor)).replace('-', '_')
    if config.DEBUG_ARRAY_OPT:
        print('gufunc_name ', type(gufunc_name), ' ', gufunc_name)
    gufunc_txt = ''
    gufunc_txt += 'def ' + gufunc_name + '(sched, ' + ', '.join(parfor_params) + '):\n'
    globls = {'np': np, 'numba': numba}
    for tup_var, exp_names in tuple_var_to_expanded_names.items():
        tup_type = typemap[tup_var]
        gufunc_txt += '    ' + param_dict[tup_var]
        if isinstance(tup_type, types.NamedTuple) or isinstance(tup_type, types.NamedUniTuple):
            named_tup = True
        else:
            named_tup = False
        if named_tup:
            func_def = guard(get_definition, lowerer.func_ir, tup_var)
            named_tuple_def = None
            if config.DEBUG_ARRAY_OPT:
                print('func_def:', func_def, type(func_def))
            if func_def is not None:
                if isinstance(func_def, ir.Expr) and func_def.op == 'call':
                    named_tuple_def = guard(get_definition, lowerer.func_ir, func_def.func)
                    if config.DEBUG_ARRAY_OPT:
                        print('named_tuple_def:', named_tuple_def, type(named_tuple_def))
                elif isinstance(func_def, ir.Arg):
                    named_tuple_def = typemap[func_def.name]
                    if config.DEBUG_ARRAY_OPT:
                        print('named_tuple_def:', named_tuple_def, type(named_tuple_def), named_tuple_def.name)
            if named_tuple_def is not None:
                if isinstance(named_tuple_def, ir.Global) or isinstance(named_tuple_def, ir.FreeVar):
                    gval = named_tuple_def.value
                    if config.DEBUG_ARRAY_OPT:
                        print('gval:', gval, type(gval))
                    globls[named_tuple_def.name] = gval
                elif isinstance(named_tuple_def, types.containers.BaseNamedTuple):
                    named_tuple_name = named_tuple_def.name.split('(')[0]
                    if config.DEBUG_ARRAY_OPT:
                        print('name:', named_tuple_name, named_tuple_def.instance_class, type(named_tuple_def.instance_class))
                    globls[named_tuple_name] = named_tuple_def.instance_class
            else:
                if config.DEBUG_ARRAY_OPT:
                    print("Didn't find definition of namedtuple for globls.")
                raise CompilerError('Could not find definition of ' + str(tup_var), tup_var.loc)
            gufunc_txt += ' = ' + tup_type.instance_class.__name__ + '('
            for name, field_name in zip(exp_names, tup_type.fields):
                gufunc_txt += field_name + '=' + param_dict[name] + ','
        else:
            gufunc_txt += ' = (' + ', '.join([param_dict[x] for x in exp_names])
            if len(exp_names) == 1:
                gufunc_txt += ','
        gufunc_txt += ')\n'
    for pindex in range(len(parfor_inputs)):
        if ascontig and isinstance(param_types[pindex], types.npytypes.Array):
            gufunc_txt += '    ' + parfor_params_orig[pindex] + ' = np.ascontiguousarray(' + parfor_params[pindex] + ')\n'
    gufunc_thread_id_var = 'ParallelAcceleratorGufuncThreadId'
    if len(parfor_redarrs) > 0:
        gufunc_txt += '    ' + gufunc_thread_id_var + ' = '
        gufunc_txt += 'numba.np.ufunc.parallel._iget_thread_id()\n'
    for arr, var in zip(parfor_redarrs, parfor_redvars):
        gufunc_txt += '    ' + param_dict[var] + '=' + param_dict[arr] + '[' + gufunc_thread_id_var + ']\n'
        if config.DEBUG_ARRAY_OPT_RUNTIME:
            gufunc_txt += '    print("thread id =", ParallelAcceleratorGufuncThreadId)\n'
            gufunc_txt += '    print("initial reduction value",ParallelAcceleratorGufuncThreadId,' + param_dict[var] + ',' + param_dict[var] + '.shape)\n'
            gufunc_txt += '    print("reduction array",ParallelAcceleratorGufuncThreadId,' + param_dict[arr] + ',' + param_dict[arr] + '.shape)\n'
    for eachdim in range(parfor_dim):
        for indent in range(eachdim + 1):
            gufunc_txt += '    '
        sched_dim = eachdim
        gufunc_txt += 'for ' + legal_loop_indices[eachdim] + ' in range(sched[' + str(sched_dim) + '], sched[' + str(sched_dim + parfor_dim) + '] + np.uint8(1)):\n'
    if config.DEBUG_ARRAY_OPT_RUNTIME:
        for indent in range(parfor_dim + 1):
            gufunc_txt += '    '
        gufunc_txt += 'print('
        for eachdim in range(parfor_dim):
            gufunc_txt += '"' + legal_loop_indices[eachdim] + '",' + legal_loop_indices[eachdim] + ','
        gufunc_txt += ')\n'
    for indent in range(parfor_dim + 1):
        gufunc_txt += '    '
    gufunc_txt += sentinel_name + ' = 0\n'
    for arr, var in zip(parfor_redarrs, parfor_redvars):
        if config.DEBUG_ARRAY_OPT_RUNTIME:
            gufunc_txt += '    print("final reduction value",ParallelAcceleratorGufuncThreadId,' + param_dict[var] + ')\n'
            gufunc_txt += '    print("final reduction array",ParallelAcceleratorGufuncThreadId,' + param_dict[arr] + ')\n'
        gufunc_txt += '    ' + param_dict[arr] + '[' + gufunc_thread_id_var + '] = ' + param_dict[var] + '\n'
    gufunc_txt += '    return None\n'
    if config.DEBUG_ARRAY_OPT:
        print('gufunc_txt = ', type(gufunc_txt), '\n', gufunc_txt)
        print('globls:', globls, type(globls))
    locls = {}
    exec(gufunc_txt, globls, locls)
    gufunc_func = locls[gufunc_name]
    if config.DEBUG_ARRAY_OPT:
        print('gufunc_func = ', type(gufunc_func), '\n', gufunc_func)
    gufunc_ir = compiler.run_frontend(gufunc_func)
    if config.DEBUG_ARRAY_OPT:
        print('gufunc_ir dump ', type(gufunc_ir))
        gufunc_ir.dump()
        print('loop_body dump ', type(loop_body))
        _print_body(loop_body)
    var_table = get_name_var_table(gufunc_ir.blocks)
    new_var_dict = {}
    reserved_names = [sentinel_name] + list(param_dict.values()) + legal_loop_indices
    for name, var in var_table.items():
        if not name in reserved_names:
            new_var_dict[name] = parfor.init_block.scope.redefine(name, loc).name
    replace_var_names(gufunc_ir.blocks, new_var_dict)
    if config.DEBUG_ARRAY_OPT:
        print('gufunc_ir dump after renaming ')
        gufunc_ir.dump()
    gufunc_param_types = [types.npytypes.Array(index_var_typ, 1, 'C')] + param_types
    if config.DEBUG_ARRAY_OPT:
        print('gufunc_param_types = ', type(gufunc_param_types), '\n', gufunc_param_types)
    gufunc_stub_last_label = find_max_label(gufunc_ir.blocks) + 1
    loop_body = add_offset_to_labels(loop_body, gufunc_stub_last_label)
    new_label = find_max_label(loop_body) + 1
    if config.DEBUG_ARRAY_OPT_RUNTIME:
        for label, block in loop_body.items():
            new_block = block.copy()
            new_block.clear()
            loc = block.loc
            scope = block.scope
            for inst in block.body:
                new_block.append(inst)
                if isinstance(inst, ir.Assign):
                    if typemap[inst.target.name] not in types.number_domain:
                        continue
                    strval = '{} ='.format(inst.target.name)
                    strconsttyp = types.StringLiteral(strval)
                    lhs = scope.redefine('str_const', loc)
                    assign_lhs = ir.Assign(value=ir.Const(value=strval, loc=loc), target=lhs, loc=loc)
                    typemap[lhs.name] = strconsttyp
                    new_block.append(assign_lhs)
                    print_node = ir.Print(args=[lhs, inst.target], vararg=None, loc=loc)
                    new_block.append(print_node)
                    sig = numba.core.typing.signature(types.none, typemap[lhs.name], typemap[inst.target.name])
                    lowerer.fndesc.calltypes[print_node] = sig
            loop_body[label] = new_block
    if config.DEBUG_ARRAY_OPT:
        print('parfor loop body')
        _print_body(loop_body)
    wrapped_blocks = wrap_loop_body(loop_body)
    hoisted, not_hoisted = hoist(parfor_params, loop_body, typemap, wrapped_blocks)
    start_block = gufunc_ir.blocks[min(gufunc_ir.blocks.keys())]
    start_block.body = start_block.body[:-1] + hoisted + [start_block.body[-1]]
    unwrap_loop_body(loop_body)
    diagnostics = lowerer.metadata['parfor_diagnostics']
    diagnostics.hoist_info[parfor.id] = {'hoisted': hoisted, 'not_hoisted': not_hoisted}
    if config.DEBUG_ARRAY_OPT:
        print('After hoisting')
        _print_body(loop_body)
    for label, block in gufunc_ir.blocks.items():
        for i, inst in enumerate(block.body):
            if isinstance(inst, ir.Assign) and inst.target.name == sentinel_name:
                loc = inst.loc
                scope = block.scope
                prev_block = ir.Block(scope, loc)
                prev_block.body = block.body[:i]
                block.body = block.body[i + 1:]
                body_first_label = min(loop_body.keys())
                prev_block.append(ir.Jump(body_first_label, loc))
                for l, b in loop_body.items():
                    gufunc_ir.blocks[l] = transfer_scope(b, scope)
                body_last_label = max(loop_body.keys())
                gufunc_ir.blocks[new_label] = block
                gufunc_ir.blocks[label] = prev_block
                gufunc_ir.blocks[body_last_label].append(ir.Jump(new_label, loc))
                break
        else:
            continue
        break
    if config.DEBUG_ARRAY_OPT:
        print('gufunc_ir last dump before renaming')
        gufunc_ir.dump()
    gufunc_ir.blocks = rename_labels(gufunc_ir.blocks)
    remove_dels(gufunc_ir.blocks)
    if config.DEBUG_ARRAY_OPT:
        print('gufunc_ir last dump')
        gufunc_ir.dump()
        print('flags', flags)
        print('typemap', typemap)
    old_alias = flags.noalias
    if not has_aliases:
        if config.DEBUG_ARRAY_OPT:
            print('No aliases found so adding noalias flag.')
        flags.noalias = True
    fixup_var_define_in_scope(gufunc_ir.blocks)

    class ParforGufuncCompiler(compiler.CompilerBase):

        def define_pipelines(self):
            from numba.core.compiler_machinery import PassManager
            dpb = compiler.DefaultPassBuilder
            pm = PassManager('full_parfor_gufunc')
            parfor_gufunc_passes = dpb.define_parfor_gufunc_pipeline(self.state)
            pm.passes.extend(parfor_gufunc_passes.passes)
            lowering_passes = dpb.define_parfor_gufunc_nopython_lowering_pipeline(self.state)
            pm.passes.extend(lowering_passes.passes)
            pm.finalize()
            return [pm]
    kernel_func = compiler.compile_ir(typingctx, targetctx, gufunc_ir, gufunc_param_types, types.none, flags, locals, pipeline_class=ParforGufuncCompiler)
    flags.noalias = old_alias
    kernel_sig = signature(types.none, *gufunc_param_types)
    if config.DEBUG_ARRAY_OPT:
        print('finished create_gufunc_for_parfor_body. kernel_sig = ', kernel_sig)
    return (kernel_func, parfor_args, kernel_sig, func_arg_types, expanded_name_to_tuple_var)