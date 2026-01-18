import itertools
import logging
import operator
import os
import time
from math import isclose
from pyomo.common.fileutils import find_library
from pyomo.common.gc_manager import PauseGC
from pyomo.opt import ProblemFormat, AbstractProblemWriter, WriterFactory
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import (
from pyomo.core.base import (
import pyomo.core.base.suffix
from pyomo.repn.standard_repn import generate_standard_repn
import pyomo.core.kernel.suffix
from pyomo.core.kernel.block import IBlock
from pyomo.core.kernel.expression import IIdentityExpression
from pyomo.core.kernel.variable import IVariable
def _print_model_NL(self, model, solver_capability, show_section_timing=False, skip_trivial_constraints=False, file_determinism=1, include_all_variable_bounds=False, export_nonlinear_variables=False):
    output_fixed_variable_bounds = self._output_fixed_variable_bounds
    symbolic_solver_labels = self._symbolic_solver_labels
    sorter = SortComponents.unsorted
    if file_determinism >= 1:
        sorter = sorter | SortComponents.indices
        if file_determinism >= 2:
            sorter = sorter | SortComponents.alphabetical
    OUTPUT = self._OUTPUT
    assert OUTPUT is not None
    overall_timer = StopWatch()
    subsection_timer = StopWatch()
    symbol_map = SymbolMap()
    name_labeler = self._name_labeler
    max_rowname_len = 0
    max_colname_len = 0
    self_ampl_var_id = self.ampl_var_id = {}
    self_ampl_con_id = self.ampl_con_id = {}
    self_ampl_obj_id = self.ampl_obj_id = {}
    Vars_dict = dict()
    Objectives_dict = dict()
    Constraints_dict = dict()
    UsedVars = set()
    LinearVars = set()
    LinearVarsInt = set()
    LinearVarsBool = set()
    self.external_byFcn = {}
    external_Libs = set()
    for fcn in model.component_objects(ExternalFunction, active=True):
        if fcn._function in self.external_byFcn:
            if self.external_byFcn[fcn._function][0]._library != fcn._library:
                raise RuntimeError('The same external function name (%s) is associated with two different libraries (%s through %s, and %s through %s).  The ASL solver will fail to link correctly.' % (fcn._function, self.external_byFcn[fcn._function]._library, self.external_byFcn[fcn._function]._library.name, fcn._library, fcn.name))
        else:
            self.external_byFcn[fcn._function] = (fcn, len(self.external_byFcn))
        external_Libs.add(fcn._library)
    if external_Libs:
        set_pyomo_amplfunc_env(external_Libs)
    elif 'PYOMO_AMPLFUNC' in os.environ:
        del os.environ['PYOMO_AMPLFUNC']
    subsection_timer.reset()
    all_blocks_list = list(model.block_data_objects(active=True, sort=sorter))
    Vars_dict = dict(enumerate(model.component_data_objects(Var, sort=sorter)))
    cntr = len(Vars_dict)
    self._varID_map = dict(((id(val), key) for key, val in Vars_dict.items()))
    self_varID_map = self._varID_map
    trivial_labeler = _Counter(cntr)
    n_objs = 0
    n_nonlinear_objs = 0
    ObjVars = set()
    ObjNonlinearVars = set()
    ObjNonlinearVarsInt = set()
    for block in all_blocks_list:
        gen_obj_repn = getattr(block, '_gen_obj_repn', None)
        if gen_obj_repn is not None:
            gen_obj_repn = bool(gen_obj_repn)
            if not hasattr(block, '_repn'):
                block._repn = ComponentMap()
            block_repn = block._repn
        for active_objective in block.component_data_objects(Objective, active=True, sort=sorter, descend_into=False):
            if symbolic_solver_labels:
                objname = name_labeler(active_objective)
                if len(objname) > max_rowname_len:
                    max_rowname_len = len(objname)
            if gen_obj_repn == False:
                repn = block_repn[active_objective]
                linear_vars = repn.linear_vars
                if repn.is_nonlinear() and repn.nonlinear_expr is None:
                    assert repn.is_quadratic()
                    assert len(repn.quadratic_vars) > 0
                    nonlinear_vars = {}
                    for v1, v2 in repn.quadratic_vars:
                        nonlinear_vars[id(v1)] = v1
                        nonlinear_vars[id(v2)] = v2
                    nonlinear_vars = nonlinear_vars.values()
                else:
                    nonlinear_vars = repn.nonlinear_vars
            else:
                repn = generate_standard_repn(active_objective.expr, quadratic=False)
                linear_vars = repn.linear_vars
                nonlinear_vars = repn.nonlinear_vars
                if gen_obj_repn:
                    block_repn[active_objective] = repn
            try:
                wrapped_repn = RepnWrapper(repn, list((self_varID_map[id(var)] for var in linear_vars)), list((self_varID_map[id(var)] for var in nonlinear_vars)))
            except KeyError as err:
                self._symbolMapKeyError(err, model, self_varID_map, list(linear_vars) + list(nonlinear_vars))
                raise
            LinearVars.update(wrapped_repn.linear_vars)
            ObjNonlinearVars.update(wrapped_repn.nonlinear_vars)
            ObjVars.update(wrapped_repn.linear_vars)
            ObjVars.update(wrapped_repn.nonlinear_vars)
            obj_ID = trivial_labeler(active_objective)
            Objectives_dict[obj_ID] = (active_objective, wrapped_repn)
            self_ampl_obj_id[obj_ID] = n_objs
            symbol_map.addSymbols([(active_objective, 'o%d' % n_objs)])
            n_objs += 1
            if repn.is_nonlinear():
                n_nonlinear_objs += 1
    if n_objs > 1:
        raise ValueError('The NL writer has detected multiple active objective functions on model %s, but currently only handles a single objective.' % model.name)
    elif n_objs == 1:
        symbol_map.alias(symbol_map.bySymbol['o0'], '__default_objective__')
    if show_section_timing:
        subsection_timer.report('Generate objective representation')
        subsection_timer.reset()
    n_ranges = 0
    n_single_sided_ineq = 0
    n_equals = 0
    n_unbounded = 0
    n_nonlinear_constraints = 0
    ConNonlinearVars = set()
    ConNonlinearVarsInt = set()
    nnz_grad_constraints = 0
    constraint_bounds_dict = {}
    nonlin_con_order_list = []
    lin_con_order_list = []
    ccons_lin = 0
    ccons_nonlin = 0
    ccons_nd = 0
    ccons_nzlb = 0
    for block in all_blocks_list:
        all_repns = list()
        gen_con_repn = getattr(block, '_gen_con_repn', None)
        if gen_con_repn is not None:
            gen_con_repn = bool(gen_con_repn)
            if not hasattr(block, '_repn'):
                block._repn = ComponentMap()
            block_repn = block._repn
        for constraint_data in block.component_data_objects(Constraint, active=True, sort=sorter, descend_into=False):
            if not constraint_data.has_lb() and (not constraint_data.has_ub()):
                assert not constraint_data.equality
                continue
            if symbolic_solver_labels:
                conname = name_labeler(constraint_data)
                if len(conname) > max_rowname_len:
                    max_rowname_len = len(conname)
            if gen_con_repn == False:
                repn = block_repn[constraint_data]
                linear_vars = repn.linear_vars
                if repn.is_nonlinear() and repn.nonlinear_expr is None:
                    assert repn.is_quadratic()
                    assert len(repn.quadratic_vars) > 0
                    nonlinear_vars = {}
                    for v1, v2 in repn.quadratic_vars:
                        nonlinear_vars[id(v1)] = v1
                        nonlinear_vars[id(v2)] = v2
                    nonlinear_vars = nonlinear_vars.values()
                else:
                    nonlinear_vars = repn.nonlinear_vars
            else:
                if constraint_data._linear_canonical_form:
                    repn = constraint_data.canonical_form()
                    linear_vars = repn.linear_vars
                    nonlinear_vars = repn.nonlinear_vars
                else:
                    repn = generate_standard_repn(constraint_data.body, quadratic=False)
                    linear_vars = repn.linear_vars
                    nonlinear_vars = repn.nonlinear_vars
                if gen_con_repn:
                    block_repn[constraint_data] = repn
            if skip_trivial_constraints and repn.is_fixed():
                continue
            con_ID = trivial_labeler(constraint_data)
            try:
                wrapped_repn = RepnWrapper(repn, list((self_varID_map[id(var)] for var in linear_vars)), list((self_varID_map[id(var)] for var in nonlinear_vars)))
            except KeyError as err:
                self._symbolMapKeyError(err, model, self_varID_map, list(linear_vars) + list(nonlinear_vars))
                raise
            if repn.is_nonlinear():
                nonlin_con_order_list.append(con_ID)
                n_nonlinear_constraints += 1
            else:
                lin_con_order_list.append(con_ID)
            Constraints_dict[con_ID] = (constraint_data, wrapped_repn)
            LinearVars.update(wrapped_repn.linear_vars)
            ConNonlinearVars.update(wrapped_repn.nonlinear_vars)
            nnz_grad_constraints += len(set(wrapped_repn.linear_vars).union(wrapped_repn.nonlinear_vars))
            L = None
            U = None
            if constraint_data.has_lb():
                L = _get_bound(constraint_data.lower)
            else:
                assert constraint_data.has_ub()
            if constraint_data.has_ub():
                U = _get_bound(constraint_data.upper)
            else:
                assert constraint_data.has_lb()
            if constraint_data.equality:
                assert L == U
            offset = repn.constant
            _type = getattr(constraint_data, '_complementarity', None)
            _vid = getattr(constraint_data, '_vid', None)
            if not _type is None:
                _vid = self_varID_map[_vid] + 1
                constraint_bounds_dict[con_ID] = '5 {0} {1}\n'.format(_type, _vid)
                if _type == 1 or _type == 2:
                    n_single_sided_ineq += 1
                elif _type == 3:
                    n_ranges += 1
                elif _type == 4:
                    n_unbounded += 1
                if repn.is_nonlinear():
                    ccons_nonlin += 1
                else:
                    ccons_lin += 1
            elif L == U:
                if L is None:
                    constraint_bounds_dict[con_ID] = '3\n'
                    n_unbounded += 1
                else:
                    constraint_bounds_dict[con_ID] = '4 %r\n' % (L - offset)
                    n_equals += 1
            elif L is None:
                constraint_bounds_dict[con_ID] = '1 %r\n' % (U - offset)
                n_single_sided_ineq += 1
            elif U is None:
                constraint_bounds_dict[con_ID] = '2 %r\n' % (L - offset)
                n_single_sided_ineq += 1
            elif L > U:
                msg = 'Constraint {0}: lower bound greater than upper bound ({1} > {2})'
                raise ValueError(msg.format(constraint_data.name, str(L), str(U)))
            else:
                constraint_bounds_dict[con_ID] = '0 %r %r\n' % (L - offset, U - offset)
                n_ranges += 1
    sos1 = solver_capability('sos1')
    sos2 = solver_capability('sos2')
    for block in all_blocks_list:
        for soscondata in block.component_data_objects(SOSConstraint, active=True, sort=sorter, descend_into=False):
            level = soscondata.level
            if level == 1 and (not sos1) or (level == 2 and (not sos2)):
                raise Exception('Solver does not support SOS level %s constraints' % (level,))
            if hasattr(soscondata, 'get_variables'):
                LinearVars.update((self_varID_map[id(vardata)] for vardata in soscondata.get_variables()))
            else:
                LinearVars.update((self_varID_map[id(vardata)] for vardata in soscondata.variables))
    self_ampl_con_id.update(((con_ID, row_id) for row_id, con_ID in enumerate(itertools.chain(nonlin_con_order_list, lin_con_order_list))))
    symbol_map.addSymbols([(Constraints_dict[con_ID][0], 'c%d' % row_id) for row_id, con_ID in enumerate(itertools.chain(nonlin_con_order_list, lin_con_order_list))])
    if show_section_timing:
        subsection_timer.report('Generate constraint representations')
        subsection_timer.reset()
    UsedVars.update(LinearVars)
    UsedVars.update(ObjNonlinearVars)
    UsedVars.update(ConNonlinearVars)
    LinearVars = LinearVars.difference(ObjNonlinearVars)
    LinearVars = LinearVars.difference(ConNonlinearVars)
    if include_all_variable_bounds:
        AllVars = set((self_varID_map[id(vardata)] for vardata in Vars_dict.values()))
        UnusedVars = AllVars.difference(UsedVars)
        LinearVars.update(UnusedVars)
    if export_nonlinear_variables:
        for v in export_nonlinear_variables:
            v_iter = v.values() if v.is_indexed() else iter((v,))
            for vi in v_iter:
                if self_varID_map[id(vi)] not in UsedVars:
                    Vars_dict[id(vi)] = vi
                    ConNonlinearVars.update([self_varID_map[id(vi)]])
    for var_ID in LinearVars:
        var = Vars_dict[var_ID]
        if var.is_binary():
            L = var.lb
            U = var.ub
            if L is None or U is None:
                raise ValueError('Variable ' + str(var.name) + 'is binary, but does not have lb and ub set')
            LinearVarsBool.add(var_ID)
        elif var.is_integer():
            LinearVarsInt.add(var_ID)
        elif not var.is_continuous():
            raise TypeError("Invalid domain type for variable with name '%s'. Variable is not continuous, integer, or binary.")
    LinearVars.difference_update(LinearVarsInt)
    LinearVars.difference_update(LinearVarsBool)
    for var_ID in ObjNonlinearVars:
        var = Vars_dict[var_ID]
        if var.is_integer() or var.is_binary():
            ObjNonlinearVarsInt.add(var_ID)
        elif not var.is_continuous():
            raise TypeError("Invalid domain type for variable with name '%s'. Variable is not continuous, integer, or binary.")
    ObjNonlinearVars.difference_update(ObjNonlinearVarsInt)
    for var_ID in ConNonlinearVars:
        var = Vars_dict[var_ID]
        if var.is_integer() or var.is_binary():
            ConNonlinearVarsInt.add(var_ID)
        elif not var.is_continuous():
            raise TypeError("Invalid domain type for variable with name '%s'. Variable is not continuous, integer, or binary.")
    ConNonlinearVars.difference_update(ConNonlinearVarsInt)
    Nonlinear_Vars_in_Objs_and_Constraints = ObjNonlinearVars.intersection(ConNonlinearVars)
    Discrete_Nonlinear_Vars_in_Objs_and_Constraints = ObjNonlinearVarsInt.intersection(ConNonlinearVarsInt)
    ObjNonlinearVars = ObjNonlinearVars.difference(Nonlinear_Vars_in_Objs_and_Constraints)
    ConNonlinearVars = ConNonlinearVars.difference(Nonlinear_Vars_in_Objs_and_Constraints)
    ObjNonlinearVarsInt = ObjNonlinearVarsInt.difference(Discrete_Nonlinear_Vars_in_Objs_and_Constraints)
    ConNonlinearVarsInt = ConNonlinearVarsInt.difference(Discrete_Nonlinear_Vars_in_Objs_and_Constraints)
    full_var_list = []
    full_var_list.extend(sorted(Nonlinear_Vars_in_Objs_and_Constraints))
    full_var_list.extend(sorted(Discrete_Nonlinear_Vars_in_Objs_and_Constraints))
    idx_nl_both = len(full_var_list)
    full_var_list.extend(sorted(ConNonlinearVars))
    full_var_list.extend(sorted(ConNonlinearVarsInt))
    idx_nl_con = len(full_var_list)
    full_var_list.extend(sorted(ObjNonlinearVars))
    full_var_list.extend(sorted(ObjNonlinearVarsInt))
    idx_nl_obj = len(full_var_list)
    full_var_list.extend(sorted(LinearVars))
    full_var_list.extend(sorted(LinearVarsBool))
    full_var_list.extend(sorted(LinearVarsInt))
    if idx_nl_obj == idx_nl_con:
        idx_nl_obj = idx_nl_both
    self_ampl_var_id.update(((var_ID, column_id) for column_id, var_ID in enumerate(full_var_list)))
    symbol_map.addSymbols([(Vars_dict[var_ID], 'v%d' % column_id) for column_id, var_ID in enumerate(full_var_list)])
    if show_section_timing:
        subsection_timer.report('Partition variable types')
        subsection_timer.reset()
    colfilename = None
    if OUTPUT.name.endswith('.nl'):
        colfilename = OUTPUT.name.replace('.nl', '.col')
    else:
        colfilename = OUTPUT.name + '.col'
    if symbolic_solver_labels:
        colf = open(colfilename, 'w')
        colfile_line_template = '%s\n'
        for var_ID in full_var_list:
            varname = name_labeler(Vars_dict[var_ID])
            colf.write(colfile_line_template % varname)
            if len(varname) > max_colname_len:
                max_colname_len = len(varname)
        colf.close()
    if show_section_timing:
        subsection_timer.report('Write .col file')
        subsection_timer.reset()
    if len(full_var_list) < 1:
        raise ValueError('No variables appear in the Pyomo model constraints or objective. This is not supported by the NL file interface')
    OUTPUT.write('g3 1 1 0\t# problem {0}\n'.format(model.name))
    OUTPUT.write(' {0} {1} {2} {3} {4} \t# vars, constraints, objectives, ranges, eqns\n'.format(len(full_var_list), n_single_sided_ineq + n_ranges + n_equals + n_unbounded, n_objs, n_ranges, n_equals))
    OUTPUT.write(' {0} {1} {2} {3} {4} {5}\t# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb\n'.format(n_nonlinear_constraints, n_nonlinear_objs, ccons_lin, ccons_nonlin, ccons_nd, ccons_nzlb))
    OUTPUT.write(' 0 0\t# network constraints: nonlinear, linear\n')
    OUTPUT.write(' {0} {1} {2} \t# nonlinear vars in constraints, objectives, both\n'.format(idx_nl_con, idx_nl_obj, idx_nl_both))
    OUTPUT.write(' 0 {0} 0 1\t# linear network variables; functions; arith, flags\n'.format(len(self.external_byFcn)))
    n_int_nonlinear_b = len(Discrete_Nonlinear_Vars_in_Objs_and_Constraints)
    n_int_nonlinear_c = len(ConNonlinearVarsInt)
    n_int_nonlinear_o = len(ObjNonlinearVarsInt)
    OUTPUT.write(' {0} {1} {2} {3} {4} \t# discrete variables: binary, integer, nonlinear (b,c,o)\n'.format(len(LinearVarsBool), len(LinearVarsInt), n_int_nonlinear_b, n_int_nonlinear_c, n_int_nonlinear_o))
    OUTPUT.write(' {0} {1} \t# nonzeros in Jacobian, obj. gradient\n'.format(nnz_grad_constraints, len(ObjVars)))
    OUTPUT.write(' %d %d\t# max name lengths: constraints, variables\n' % (max_rowname_len, max_colname_len))
    OUTPUT.write(' 0 0 0 0 0\t# common exprs: b,c,o,c1,o1\n')
    for fcn, fid in sorted(self.external_byFcn.values(), key=operator.itemgetter(1)):
        OUTPUT.write('F%d 1 -1 %s\n' % (fid, fcn._function))
    sos1 = solver_capability('sos1')
    sos2 = solver_capability('sos2')
    modelSOS = ModelSOS(self_ampl_var_id, self_varID_map)
    for block in all_blocks_list:
        for soscondata in block.component_data_objects(SOSConstraint, active=True, sort=sorter, descend_into=False):
            level = soscondata.level
            if level == 1 and (not sos1) or (level == 2 and (not sos2)):
                raise ValueError('Solver does not support SOS level %s constraints' % level)
            modelSOS.count_constraint(soscondata)
    symbol_map_byObject = symbol_map.byObject
    var_sosno_suffix = modelSOS.sosno
    var_ref_suffix = modelSOS.ref
    sosconstraint_sosno_vals = set(var_sosno_suffix.vals)
    suffix_header_line = 'S{0} {1} {2}\n'
    suffix_line = '{0} {1!r}\n'
    var_tag = 0
    con_tag = 1
    obj_tag = 2
    prob_tag = 3
    suffix_dict = {}
    if isinstance(model, IBlock):
        suffix_gen = lambda b: ((suf.storage_key, suf) for suf in pyomo.core.kernel.suffix.export_suffix_generator(b, active=True, descend_into=False))
    else:
        suffix_gen = lambda b: pyomo.core.base.suffix.active_export_suffix_generator(b)
    for block in all_blocks_list:
        for name, suf in suffix_gen(block):
            if len(suf):
                suffix_dict.setdefault(name, []).append(suf)
    if not 'sosno' in suffix_dict:
        s_lines = var_sosno_suffix.genfilelines()
        len_s_lines = len(s_lines)
        if len_s_lines > 0:
            OUTPUT.write(suffix_header_line.format(var_tag, len_s_lines, 'sosno'))
            OUTPUT.writelines(s_lines)
    elif not var_sosno_suffix.is_empty():
        raise RuntimeError("The Pyomo NL file writer does not allow both manually declared 'sosno' suffixes as well as SOSConstraint components to exist on a single model. To avoid this error please use only one of these methods to define special ordered sets.")
    if not 'ref' in suffix_dict:
        s_lines = var_ref_suffix.genfilelines()
        len_s_lines = len(s_lines)
        if len_s_lines > 0:
            OUTPUT.write(suffix_header_line.format(var_tag, len_s_lines, 'ref'))
            OUTPUT.writelines(s_lines)
    elif not var_ref_suffix.is_empty():
        raise RuntimeError("The Pyomo NL file writer does not allow both manually declared 'ref' suffixes as well as SOSConstraint components to exist on a single model. To avoid this error please use only one of these methods to define special ordered sets.")
    for suffix_name in sorted(suffix_dict):
        suffixes = suffix_dict[suffix_name]
        datatypes = set()
        for suffix in suffixes:
            try:
                datatype = suffix.datatype
            except AttributeError:
                datatype = suffix.get_datatype()
            if datatype not in (Suffix.FLOAT, Suffix.INT):
                raise ValueError('The Pyomo NL file writer requires that all active export Suffix components declare a numeric datatype. Suffix component: %s with ' % suffix_name)
            datatypes.add(datatype)
        if len(datatypes) != 1:
            raise ValueError('The Pyomo NL file writer found multiple active export suffix components with name %s with different datatypes. A single datatype must be declared.' % suffix_name)
        if suffix_name == 'dual':
            continue
        float_tag = 0
        if datatypes.pop() == Suffix.FLOAT:
            float_tag = 4
        var_s_lines = []
        con_s_lines = []
        obj_s_lines = []
        mod_s_lines = []
        for suffix in suffixes:
            for component_data, suffix_value in suffix.items():
                try:
                    symbol = symbol_map_byObject[id(component_data)]
                    type_tag = symbol[0]
                    ampl_id = int(symbol[1:])
                    if type_tag == 'v':
                        var_s_lines.append((ampl_id, suffix_value))
                    elif type_tag == 'c':
                        con_s_lines.append((ampl_id, suffix_value))
                    elif type_tag == 'o':
                        obj_s_lines.append((ampl_id, suffix_value))
                    else:
                        assert False
                except KeyError:
                    if component_data is model:
                        mod_s_lines.append((0, suffix_value))
        if len(var_s_lines) > 0:
            OUTPUT.write(suffix_header_line.format(var_tag | float_tag, len(var_s_lines), suffix_name))
            OUTPUT.writelines((suffix_line.format(*_l) for _l in sorted(var_s_lines, key=operator.itemgetter(0))))
        if len(con_s_lines) > 0:
            OUTPUT.write(suffix_header_line.format(con_tag | float_tag, len(con_s_lines), suffix_name))
            OUTPUT.writelines((suffix_line.format(*_l) for _l in sorted(con_s_lines, key=operator.itemgetter(0))))
        if len(obj_s_lines) > 0:
            OUTPUT.write(suffix_header_line.format(obj_tag | float_tag, len(obj_s_lines), suffix_name))
            OUTPUT.writelines((suffix_line.format(*_l) for _l in sorted(obj_s_lines, key=operator.itemgetter(0))))
        if len(mod_s_lines) > 0:
            if len(mod_s_lines) > 1:
                logger.warning('ProblemWriter_nl: Collected multiple values for Suffix %s referencing model %s. This is likely a bug.' % (suffix_name, model.name))
            OUTPUT.write(suffix_header_line.format(prob_tag | float_tag, len(mod_s_lines), suffix_name))
            OUTPUT.writelines((suffix_line.format(*_l) for _l in sorted(mod_s_lines, key=operator.itemgetter(0))))
    del modelSOS
    rowfilename = None
    if OUTPUT.name.endswith('.nl'):
        rowfilename = OUTPUT.name.replace('.nl', '.row')
    else:
        rowfilename = OUTPUT.name + '.row'
    if symbolic_solver_labels:
        rowf = open(rowfilename, 'w')
    cu = [0 for i in range(len(full_var_list))]
    for con_ID in nonlin_con_order_list:
        con_data, wrapped_repn = Constraints_dict[con_ID]
        row_id = self_ampl_con_id[con_ID]
        OUTPUT.write('C%d' % row_id)
        if symbolic_solver_labels:
            lbl = name_labeler(con_data)
            OUTPUT.write('\t#%s' % lbl)
            rowf.write(lbl + '\n')
        OUTPUT.write('\n')
        if wrapped_repn.repn.nonlinear_expr is not None:
            assert not wrapped_repn.repn.is_quadratic()
            self._print_nonlinear_terms_NL(wrapped_repn.repn.nonlinear_expr)
        else:
            assert wrapped_repn.repn.is_quadratic()
            self._print_standard_quadratic_NL(wrapped_repn.repn.quadratic_vars, wrapped_repn.repn.quadratic_coefs)
        for var_ID in set(wrapped_repn.linear_vars).union(wrapped_repn.nonlinear_vars):
            cu[self_ampl_var_id[var_ID]] += 1
    for con_ID in lin_con_order_list:
        con_data, wrapped_repn = Constraints_dict[con_ID]
        row_id = self_ampl_con_id[con_ID]
        con_vars = set(wrapped_repn.linear_vars)
        for var_ID in con_vars:
            cu[self_ampl_var_id[var_ID]] += 1
        OUTPUT.write('C%d' % row_id)
        if symbolic_solver_labels:
            lbl = name_labeler(con_data)
            OUTPUT.write('\t#%s' % lbl)
            rowf.write(lbl + '\n')
        OUTPUT.write('\n')
        OUTPUT.write('n0\n')
    if show_section_timing:
        subsection_timer.report('Write NL header and suffix lines')
        subsection_timer.reset()
    for obj_ID, (obj, wrapped_repn) in Objectives_dict.items():
        k = 0
        if not obj.is_minimizing():
            k = 1
        OUTPUT.write('O%d %d' % (self_ampl_obj_id[obj_ID], k))
        if symbolic_solver_labels:
            lbl = name_labeler(obj)
            OUTPUT.write('\t#%s' % lbl)
            rowf.write(lbl + '\n')
        OUTPUT.write('\n')
        if wrapped_repn.repn.is_linear():
            OUTPUT.write(self._op_string[NumericConstant] % wrapped_repn.repn.constant)
        else:
            if wrapped_repn.repn.constant != 0:
                _, binary_sum_str, _ = self._op_string[EXPR.SumExpressionBase]
                OUTPUT.write(binary_sum_str)
                OUTPUT.write(self._op_string[NumericConstant] % wrapped_repn.repn.constant)
            if wrapped_repn.repn.nonlinear_expr is not None:
                assert not wrapped_repn.repn.is_quadratic()
                self._print_nonlinear_terms_NL(wrapped_repn.repn.nonlinear_expr)
            else:
                assert wrapped_repn.repn.is_quadratic()
                self._print_standard_quadratic_NL(wrapped_repn.repn.quadratic_vars, wrapped_repn.repn.quadratic_coefs)
    if symbolic_solver_labels:
        rowf.close()
    del name_labeler
    if show_section_timing:
        subsection_timer.report('Write objective expression')
        subsection_timer.reset()
    if 'dual' in suffix_dict:
        s_lines = []
        for dual_suffix in suffix_dict['dual']:
            for constraint_data, suffix_value in dual_suffix.items():
                try:
                    symbol = symbol_map_byObject[id(constraint_data)]
                    type_tag = symbol[0]
                    assert type_tag == 'c'
                    ampl_con_id = int(symbol[1:])
                    s_lines.append((ampl_con_id, suffix_value))
                except KeyError:
                    pass
        if len(s_lines) > 0:
            OUTPUT.write('d%d' % len(s_lines))
            if symbolic_solver_labels:
                OUTPUT.write('\t# dual initial guess')
            OUTPUT.write('\n')
            OUTPUT.writelines((suffix_line.format(*_l) for _l in sorted(s_lines, key=operator.itemgetter(0))))
    var_bound_list = []
    x_init_list = []
    for ampl_var_id, var_ID in enumerate(full_var_list):
        var = Vars_dict[var_ID]
        if var.value is not None:
            x_init_list.append('%d %r\n' % (ampl_var_id, var.value))
        if var.fixed:
            if not output_fixed_variable_bounds:
                raise ValueError("Encountered a fixed variable (%s) inside an active objective or constraint expression on model %s, which is usually indicative of a preprocessing error. Use the IO-option 'output_fixed_variable_bounds=True' to suppress this error and fix the variable by overwriting its bounds in the NL file." % (var.name, model.name))
            if var.value is None:
                raise ValueError('Variable cannot be fixed to a value of None.')
            L = U = _get_bound(var.value)
        else:
            L = None
            if var.has_lb():
                L = _get_bound(var.lb)
            U = None
            if var.has_ub():
                U = _get_bound(var.ub)
        if L is not None:
            if U is not None:
                if L == U:
                    var_bound_list.append('4 %r\n' % L)
                else:
                    var_bound_list.append('0 %r %r\n' % (L, U))
            else:
                var_bound_list.append('2 %r\n' % L)
        elif U is not None:
            var_bound_list.append('1 %r\n' % U)
        else:
            var_bound_list.append('3\n')
    OUTPUT.write('x%d' % len(x_init_list))
    if symbolic_solver_labels:
        OUTPUT.write('\t# initial guess')
    OUTPUT.write('\n')
    OUTPUT.writelines(x_init_list)
    del x_init_list
    if show_section_timing:
        subsection_timer.report('Write initializations')
        subsection_timer.reset()
    OUTPUT.write('r')
    if symbolic_solver_labels:
        OUTPUT.write("\t#%d ranges (rhs's)" % (len(nonlin_con_order_list) + len(lin_con_order_list)))
    OUTPUT.write('\n')
    OUTPUT.writelines((constraint_bounds_dict[con_ID] for con_ID in itertools.chain(nonlin_con_order_list, lin_con_order_list)))
    if show_section_timing:
        subsection_timer.report('Write constraint bounds')
        subsection_timer.reset()
    OUTPUT.write('b')
    if symbolic_solver_labels:
        OUTPUT.write('\t#%d bounds (on variables)' % len(var_bound_list))
    OUTPUT.write('\n')
    OUTPUT.writelines(var_bound_list)
    del var_bound_list
    if show_section_timing:
        subsection_timer.report('Write variable bounds')
        subsection_timer.reset()
    ktot = 0
    n1 = len(full_var_list) - 1
    OUTPUT.write('k%d' % n1)
    if symbolic_solver_labels:
        OUTPUT.write('\t#intermediate Jacobian column lengths')
    OUTPUT.write('\n')
    ktot = 0
    for i in range(n1):
        ktot += cu[i]
        OUTPUT.write('%d\n' % ktot)
    del cu
    if show_section_timing:
        subsection_timer.report('Write k lines')
        subsection_timer.reset()
    for nc, con_ID in enumerate(itertools.chain(nonlin_con_order_list, lin_con_order_list)):
        con_data, wrapped_repn = Constraints_dict[con_ID]
        numnonlinear_vars = len(wrapped_repn.nonlinear_vars)
        numlinear_vars = len(wrapped_repn.linear_vars)
        if numnonlinear_vars == 0:
            if numlinear_vars > 0:
                linear_dict = dict(((var_ID, coef) for var_ID, coef in zip(wrapped_repn.linear_vars, wrapped_repn.repn.linear_coefs)))
                OUTPUT.write('J%d %d\n' % (nc, numlinear_vars))
                OUTPUT.writelines(('%d %r\n' % (self_ampl_var_id[con_var], linear_dict[con_var]) for con_var in sorted(linear_dict.keys())))
        elif numlinear_vars == 0:
            nl_con_vars = sorted(wrapped_repn.nonlinear_vars)
            OUTPUT.write('J%d %d\n' % (nc, numnonlinear_vars))
            OUTPUT.writelines(('%d 0\n' % self_ampl_var_id[con_var] for con_var in nl_con_vars))
        else:
            con_vars = set(wrapped_repn.nonlinear_vars)
            nl_con_vars = sorted(con_vars.difference(wrapped_repn.linear_vars))
            con_vars.update(wrapped_repn.linear_vars)
            linear_dict = dict(((var_ID, coef) for var_ID, coef in zip(wrapped_repn.linear_vars, wrapped_repn.repn.linear_coefs)))
            OUTPUT.write('J%d %d\n' % (nc, len(con_vars)))
            OUTPUT.writelines(('%d %r\n' % (self_ampl_var_id[con_var], linear_dict[con_var]) for con_var in sorted(linear_dict.keys())))
            OUTPUT.writelines(('%d 0\n' % self_ampl_var_id[con_var] for con_var in nl_con_vars))
    if show_section_timing:
        subsection_timer.report('Write J lines')
        subsection_timer.reset()
    for obj_ID, (obj, wrapped_repn) in Objectives_dict.items():
        grad_entries = {}
        for idx, obj_var in enumerate(wrapped_repn.linear_vars):
            grad_entries[self_ampl_var_id[obj_var]] = wrapped_repn.repn.linear_coefs[idx]
        for obj_var in wrapped_repn.nonlinear_vars:
            if obj_var not in wrapped_repn.linear_vars:
                grad_entries[self_ampl_var_id[obj_var]] = 0
        len_ge = len(grad_entries)
        if len_ge > 0:
            OUTPUT.write('G%d %d\n' % (self_ampl_obj_id[obj_ID], len_ge))
            for var_ID in sorted(grad_entries.keys()):
                OUTPUT.write('%d %r\n' % (var_ID, grad_entries[var_ID]))
    if show_section_timing:
        subsection_timer.report('Write G lines')
        subsection_timer.reset()
        overall_timer.report('Total time')
    return symbol_map