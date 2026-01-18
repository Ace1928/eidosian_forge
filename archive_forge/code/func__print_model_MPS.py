import logging
from io import StringIO
from pyomo.common.gc_manager import PauseGC
from pyomo.opt import ProblemFormat
from pyomo.opt.base import AbstractProblemWriter, WriterFactory
from pyomo.core.base import (
from pyomo.repn import generate_standard_repn
def _print_model_MPS(self, model, output_file, solver_capability, labeler, output_fixed_variable_bounds=False, file_determinism=1, row_order=None, column_order=None, skip_trivial_constraints=False, force_objective_constant=False, include_all_variable_bounds=False, skip_objective_sense=False):
    symbol_map = SymbolMap()
    variable_symbol_map = SymbolMap()
    extract_variable_coefficients = self._extract_variable_coefficients
    create_symbol_func = SymbolMap.createSymbol
    create_symbols_func = SymbolMap.createSymbols
    alias_symbol_func = SymbolMap.alias
    variable_label_pairs = []
    sortOrder = SortComponents.unsorted
    if file_determinism >= 1:
        sortOrder = sortOrder | SortComponents.indices
        if file_determinism >= 2:
            sortOrder = sortOrder | SortComponents.alphabetical
    all_blocks = []
    variable_list = []
    for block in model.block_data_objects(active=True, sort=sortOrder):
        all_blocks.append(block)
        for vardata in block.component_data_objects(Var, active=True, sort=sortOrder, descend_into=False):
            variable_list.append(vardata)
            variable_label_pairs.append((vardata, create_symbol_func(symbol_map, vardata, labeler)))
    variable_symbol_map.addSymbols(variable_label_pairs)
    object_symbol_dictionary = symbol_map.byObject
    variable_symbol_dictionary = variable_symbol_map.byObject
    if column_order is not None:
        variable_list.sort(key=lambda _x: column_order[_x])
    variable_to_column = ComponentMap(((vardata, i) for i, vardata in enumerate(variable_list)))
    column_data = [[] for i in range(len(variable_list) + 1)]
    quadobj_data = []
    quadmatrix_data = []
    rhs_data = []
    output_file.write('* Source:     Pyomo MPS Writer\n')
    output_file.write('* Format:     Free MPS\n')
    output_file.write('*\n')
    output_file.write('NAME %s\n' % (model.name,))
    objective_label = None
    numObj = 0
    onames = []
    for block in all_blocks:
        gen_obj_repn = getattr(block, '_gen_obj_repn', True)
        if not hasattr(block, '_repn'):
            block._repn = ComponentMap()
        block_repn = block._repn
        for objective_data in block.component_data_objects(Objective, active=True, sort=sortOrder, descend_into=False):
            numObj += 1
            onames.append(objective_data.name)
            if numObj > 1:
                raise ValueError("More than one active objective defined for input model '%s'; Cannot write legal MPS file\nObjectives: %s" % (model.name, ' '.join(onames)))
            objective_label = create_symbol_func(symbol_map, objective_data, labeler)
            symbol_map.alias(objective_data, '__default_objective__')
            if not skip_objective_sense:
                output_file.write('OBJSENSE\n')
                if objective_data.is_minimizing():
                    output_file.write(' MIN\n')
                else:
                    output_file.write(' MAX\n')
            output_file.write('ROWS\n')
            output_file.write(' N  %s\n' % objective_label)
            if gen_obj_repn:
                repn = generate_standard_repn(objective_data.expr)
                block_repn[objective_data] = repn
            else:
                repn = block_repn[objective_data]
            degree = repn.polynomial_degree()
            if degree == 0:
                logger.warning('Constant objective detected, replacing with a placeholder to prevent solver failure.')
                force_objective_constant = True
            elif degree is None:
                raise RuntimeError("Cannot write legal MPS file. Objective '%s' has nonlinear terms that are not quadratic." % objective_data.name)
            constant = extract_variable_coefficients(objective_label, repn, column_data, quadobj_data, variable_to_column)
            if force_objective_constant or constant != 0.0:
                column_data[-1].append((objective_label, constant))
    if numObj == 0:
        raise ValueError("Cannot write legal MPS file: No objective defined for input model '%s'." % str(model))
    assert objective_label is not None

    def constraint_generator():
        for block in all_blocks:
            gen_con_repn = getattr(block, '_gen_con_repn', True)
            if not hasattr(block, '_repn'):
                block._repn = ComponentMap()
            block_repn = block._repn
            for constraint_data in block.component_data_objects(Constraint, active=True, sort=sortOrder, descend_into=False):
                if not constraint_data.has_lb() and (not constraint_data.has_ub()):
                    assert not constraint_data.equality
                    continue
                if constraint_data._linear_canonical_form:
                    repn = constraint_data.canonical_form()
                elif gen_con_repn:
                    repn = generate_standard_repn(constraint_data.body)
                    block_repn[constraint_data] = repn
                else:
                    repn = block_repn[constraint_data]
                yield (constraint_data, repn)
    if row_order is not None:
        sorted_constraint_list = list(constraint_generator())
        sorted_constraint_list.sort(key=lambda x: row_order[x[0]])

        def yield_all_constraints():
            for constraint_data, repn in sorted_constraint_list:
                yield (constraint_data, repn)
    else:
        yield_all_constraints = constraint_generator
    for constraint_data, repn in yield_all_constraints():
        degree = repn.polynomial_degree()
        if degree == 0:
            if skip_trivial_constraints:
                continue
        elif degree is None:
            raise RuntimeError("Cannot write legal MPS file. Constraint '%s' has nonlinear terms that are not quadratic." % constraint_data.name)
        con_symbol = create_symbol_func(symbol_map, constraint_data, labeler)
        if constraint_data.equality:
            assert value(constraint_data.lower) == value(constraint_data.upper)
            label = 'c_e_' + con_symbol + '_'
            alias_symbol_func(symbol_map, constraint_data, label)
            output_file.write(' E  %s\n' % label)
            offset = extract_variable_coefficients(label, repn, column_data, quadmatrix_data, variable_to_column)
            bound = constraint_data.lower
            bound = _get_bound(bound) - offset
            rhs_data.append((label, _no_negative_zero(bound)))
        else:
            if constraint_data.has_lb():
                if constraint_data.has_ub():
                    label = 'r_l_' + con_symbol + '_'
                else:
                    label = 'c_l_' + con_symbol + '_'
                alias_symbol_func(symbol_map, constraint_data, label)
                output_file.write(' G  %s\n' % label)
                offset = extract_variable_coefficients(label, repn, column_data, quadmatrix_data, variable_to_column)
                bound = constraint_data.lower
                bound = _get_bound(bound) - offset
                rhs_data.append((label, _no_negative_zero(bound)))
            else:
                assert constraint_data.has_ub()
            if constraint_data.has_ub():
                if constraint_data.has_lb():
                    label = 'r_u_' + con_symbol + '_'
                else:
                    label = 'c_u_' + con_symbol + '_'
                alias_symbol_func(symbol_map, constraint_data, label)
                output_file.write(' L  %s\n' % label)
                offset = extract_variable_coefficients(label, repn, column_data, quadmatrix_data, variable_to_column)
                bound = constraint_data.upper
                bound = _get_bound(bound) - offset
                rhs_data.append((label, _no_negative_zero(bound)))
            else:
                assert constraint_data.has_lb()
    if len(column_data[-1]) > 0:
        output_file.write(' E  c_e_ONE_VAR_CONSTANT\n')
        column_data[-1].append(('c_e_ONE_VAR_CONSTANT', 1))
        rhs_data.append(('c_e_ONE_VAR_CONSTANT', 1))
    column_template = '     %s %s %' + self._precision_string + '\n'
    output_file.write('COLUMNS\n')
    cnt = 0
    in_integer_section = False
    mark_cnt = 0
    for vardata in variable_list:
        col_entries = column_data[variable_to_column[vardata]]
        cnt += 1
        if len(col_entries) > 0:
            if self._int_marker:
                if vardata.is_integer():
                    if not in_integer_section:
                        output_file.write(f"     MARK{mark_cnt:04d} 'MARKER' 'INTORG'\n")
                        in_integer_section = True
                        mark_cnt += 1
                elif in_integer_section:
                    output_file.write(f"     MARK{mark_cnt:04d} 'MARKER' 'INTEND'\n")
                    in_integer_section = False
                    mark_cnt += 1
            var_label = variable_symbol_dictionary[id(vardata)]
            for i, (row_label, coef) in enumerate(col_entries):
                output_file.write(column_template % (var_label, row_label, _no_negative_zero(coef)))
        elif include_all_variable_bounds:
            var_label = variable_symbol_dictionary[id(vardata)]
            output_file.write(column_template % (var_label, objective_label, 0))
    if self._int_marker and in_integer_section:
        output_file.write(f"     MARK{mark_cnt:04d} 'MARKER' 'INTEND'\n")
    assert cnt == len(column_data) - 1
    if len(column_data[-1]) > 0:
        col_entries = column_data[-1]
        var_label = 'ONE_VAR_CONSTANT'
        for i, (row_label, coef) in enumerate(col_entries):
            output_file.write(column_template % (var_label, row_label, _no_negative_zero(coef)))
    rhs_template = '     RHS %s %' + self._precision_string + '\n'
    output_file.write('RHS\n')
    for i, (row_label, rhs) in enumerate(rhs_data):
        output_file.write(rhs_template % (row_label, rhs))
    SOSlines = StringIO()
    sos1 = solver_capability('sos1')
    sos2 = solver_capability('sos2')
    for block in all_blocks:
        for soscondata in block.component_data_objects(SOSConstraint, active=True, sort=sortOrder, descend_into=False):
            create_symbol_func(symbol_map, soscondata, labeler)
            level = soscondata.level
            if level == 1 and (not sos1) or (level == 2 and (not sos2)) or level > 2:
                raise ValueError('Solver does not support SOS level %s constraints' % level)
            self._printSOS(symbol_map, labeler, variable_symbol_map, soscondata, SOSlines)
    entry_template = '%s %' + self._precision_string + '\n'
    output_file.write('BOUNDS\n')
    for vardata in variable_list:
        if include_all_variable_bounds or id(vardata) in self._referenced_variable_ids:
            var_label = variable_symbol_dictionary[id(vardata)]
            if vardata.fixed:
                if not output_fixed_variable_bounds:
                    raise ValueError("Encountered a fixed variable (%s) inside an active objective or constraint expression on model %s, which is usually indicative of a preprocessing error. Use the IO-option 'output_fixed_variable_bounds=True' to suppress this error and fix the variable by overwriting its bounds in the MPS file." % (vardata.name, model.name))
                if vardata.value is None:
                    raise ValueError('Variable cannot be fixed to a value of None.')
                output_file.write((' FX BOUND ' + entry_template) % (var_label, _no_negative_zero(value(vardata.value))))
                continue
            vardata_lb = _no_negative_zero(_get_bound(vardata.lb))
            vardata_ub = _no_negative_zero(_get_bound(vardata.ub))
            unbounded_lb = not vardata.has_lb()
            unbounded_ub = not vardata.has_ub()
            treat_as_integer = False
            if vardata.is_binary():
                if vardata_lb == 0 and vardata_ub == 1:
                    output_file.write(' BV BOUND %s\n' % var_label)
                    continue
                else:
                    treat_as_integer = True
            if treat_as_integer or vardata.is_integer():
                if not unbounded_lb:
                    output_file.write((' LI BOUND ' + entry_template) % (var_label, vardata_lb))
                else:
                    output_file.write(' LI BOUND %s -10E20\n' % var_label)
                if not unbounded_ub:
                    output_file.write((' UI BOUND ' + entry_template) % (var_label, vardata_ub))
                else:
                    output_file.write(' UI BOUND %s 10E20\n' % var_label)
            else:
                assert vardata.is_continuous()
                if unbounded_lb and unbounded_ub:
                    output_file.write(' FR BOUND %s\n' % var_label)
                else:
                    if not unbounded_lb:
                        output_file.write((' LO BOUND ' + entry_template) % (var_label, vardata_lb))
                    else:
                        output_file.write(' MI BOUND %s\n' % var_label)
                    if not unbounded_ub:
                        output_file.write((' UP BOUND ' + entry_template) % (var_label, vardata_ub))
    output_file.write(SOSlines.getvalue())
    if len(quadobj_data) > 0:
        assert len(quadobj_data) == 1
        output_file.write('QUADOBJ\n')
        label, quad_terms = quadobj_data[0]
        assert label == objective_label
        quad_terms = sorted(quad_terms, key=lambda _x: sorted((variable_to_column[_x[0][0]], variable_to_column[_x[0][1]])))
        for term, coef in quad_terms:
            var1, var2 = sorted(term, key=lambda _x: variable_to_column[_x])
            var1_label = variable_symbol_dictionary[id(var1)]
            var2_label = variable_symbol_dictionary[id(var2)]
            if var1_label == var2_label:
                output_file.write(column_template % (var1_label, var2_label, _no_negative_zero(coef * 2)))
            else:
                output_file.write(column_template % (var1_label, var2_label, _no_negative_zero(coef)))
                output_file.write(column_template % (var2_label, var1_label, _no_negative_zero(coef)))
    if len(quadmatrix_data) > 0:
        for row_label, quad_terms in quadmatrix_data:
            output_file.write('QCMATRIX    %s\n' % row_label)
            quad_terms = sorted(quad_terms, key=lambda _x: sorted((variable_to_column[_x[0][0]], variable_to_column[_x[0][1]])))
            for term, coef in quad_terms:
                var1, var2 = sorted(term, key=lambda _x: variable_to_column[_x])
                var1_label = variable_symbol_dictionary[id(var1)]
                var2_label = variable_symbol_dictionary[id(var2)]
                if var1_label == var2_label:
                    output_file.write(column_template % (var1_label, var2_label, _no_negative_zero(coef)))
                else:
                    output_file.write(column_template % (var1_label, var2_label, _no_negative_zero(coef * 0.5)))
                    output_file.write(column_template % (var2_label, var1_label, coef * 0.5))
    output_file.write('ENDATA\n')
    vars_to_delete = set(variable_symbol_map.byObject.keys()) - set(self._referenced_variable_ids.keys())
    sm_byObject = symbol_map.byObject
    sm_bySymbol = symbol_map.bySymbol
    var_sm_byObject = variable_symbol_map.byObject
    for varid in vars_to_delete:
        symbol = var_sm_byObject[varid]
        del sm_byObject[varid]
        del sm_bySymbol[symbol]
    del variable_symbol_map
    return symbol_map