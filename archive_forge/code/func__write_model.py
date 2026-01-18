from io import StringIO
from pyomo.common.gc_manager import PauseGC
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.visitor import _ToStringVisitor
from pyomo.core.base import (
from pyomo.core.base.component import ActiveComponent
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.opt import ProblemFormat
from pyomo.opt.base import AbstractProblemWriter, WriterFactory
from pyomo.repn.util import valid_expr_ctypes_minlp, valid_active_ctypes_minlp, ftoa
import logging
def _write_model(self, model, output_file, solver_capability, var_list, var_label, symbolMap, con_labeler, sort, skip_trivial_constraints, output_fixed_variables, warmstart, solver, mtype, solprint, limrow, limcol, solvelink, add_options, put_results, put_results_format):
    constraint_names = []
    ConstraintIO = StringIO()
    linear = True
    linear_degree = set([0, 1])
    dnlp = False
    model_ctypes = model.collect_ctypes(active=True)
    invalids = set()
    for t in model_ctypes - valid_active_ctypes_minlp:
        if issubclass(t, ActiveComponent):
            invalids.add(t)
    if len(invalids):
        invalids = [t.__name__ for t in invalids]
        raise RuntimeError('Unallowable active component(s) %s.\nThe GAMS writer cannot export models with this component type.' % ', '.join(invalids))
    tc = StorageTreeChecker(model)
    for con in model.component_data_objects(Constraint, active=True, sort=sort):
        if not con.has_lb() and (not con.has_ub()):
            assert not con.equality
            continue
        con_body = as_numeric(con.body)
        if skip_trivial_constraints and con_body.is_fixed():
            continue
        if linear:
            if con_body.polynomial_degree() not in linear_degree:
                linear = False
        cName = symbolMap.getSymbol(con, con_labeler)
        con_body_str, con_discontinuous = expression_to_string(con_body, tc, smap=symbolMap, output_fixed_variables=output_fixed_variables)
        dnlp |= con_discontinuous
        if con.equality:
            constraint_names.append('%s' % cName)
            ConstraintIO.write('%s.. %s =e= %s ;\n' % (constraint_names[-1], con_body_str, ftoa(con.upper, False)))
        else:
            if con.has_lb():
                constraint_names.append('%s_lo' % cName)
                ConstraintIO.write('%s.. %s =l= %s ;\n' % (constraint_names[-1], ftoa(con.lower, False), con_body_str))
            if con.has_ub():
                constraint_names.append('%s_hi' % cName)
                ConstraintIO.write('%s.. %s =l= %s ;\n' % (constraint_names[-1], con_body_str, ftoa(con.upper, False)))
    obj = list(model.component_data_objects(Objective, active=True, sort=sort))
    if len(obj) != 1:
        raise RuntimeError('GAMS writer requires exactly one active objective (found %s)' % len(obj))
    obj = obj[0]
    if linear:
        if obj.polynomial_degree() not in linear_degree:
            linear = False
    obj_expr_str, obj_discontinuous = expression_to_string(obj.expr, tc, smap=symbolMap, output_fixed_variables=output_fixed_variables)
    dnlp |= obj_discontinuous
    oName = symbolMap.getSymbol(obj, con_labeler)
    constraint_names.append(oName)
    ConstraintIO.write('%s.. GAMS_OBJECTIVE =e= %s ;\n' % (oName, obj_expr_str))
    categorized_vars = Categorizer(var_list, symbolMap)
    output_file.write('$offlisting\n')
    output_file.write('$offdigit\n\n')
    output_file.write('EQUATIONS\n\t')
    output_file.write('\n\t'.join(constraint_names))
    if categorized_vars.binary:
        output_file.write(';\n\nBINARY VARIABLES\n\t')
        output_file.write('\n\t'.join(categorized_vars.binary))
    if categorized_vars.ints:
        output_file.write(';\n\nINTEGER VARIABLES')
        output_file.write('\n\t')
        output_file.write('\n\t'.join(categorized_vars.ints))
    if categorized_vars.positive:
        output_file.write(';\n\nPOSITIVE VARIABLES\n\t')
        output_file.write('\n\t'.join(categorized_vars.positive))
    output_file.write(';\n\nVARIABLES\n\tGAMS_OBJECTIVE\n\t')
    output_file.write('\n\t'.join(categorized_vars.reals + categorized_vars.fixed))
    output_file.write(';\n\n')
    for var in categorized_vars.fixed:
        output_file.write('%s.fx = %s;\n' % (var, ftoa(value(symbolMap.getObject(var)), False)))
    output_file.write('\n')
    for line in ConstraintIO.getvalue().splitlines():
        if len(line) > 80000:
            line = split_long_line(line)
        output_file.write(line + '\n')
    output_file.write('\n')
    warn_int_bounds = False
    for category, var_name in categorized_vars:
        var = symbolMap.getObject(var_name)
        tc(var)
        lb, ub = var.bounds
        if category == 'positive':
            if ub is not None:
                output_file.write('%s.up = %s;\n' % (var_name, ftoa(ub, False)))
        elif category == 'ints':
            if lb is None:
                warn_int_bounds = True
                logger.warning('Lower bound for integer variable %s set to -1.0E+100.' % var.name)
                output_file.write('%s.lo = -1.0E+100;\n' % var_name)
            elif lb != 0:
                output_file.write('%s.lo = %s;\n' % (var_name, ftoa(lb, False)))
            if ub is None:
                warn_int_bounds = True
                logger.warning('Upper bound for integer variable %s set to +1.0E+100.' % var.name)
                output_file.write('%s.up = +1.0E+100;\n' % var_name)
            else:
                output_file.write('%s.up = %s;\n' % (var_name, ftoa(ub, False)))
        elif category == 'binary':
            if lb != 0:
                output_file.write('%s.lo = %s;\n' % (var_name, ftoa(lb, False)))
            if ub != 1:
                output_file.write('%s.up = %s;\n' % (var_name, ftoa(ub, False)))
        elif category == 'reals':
            if lb is not None:
                output_file.write('%s.lo = %s;\n' % (var_name, ftoa(lb, False)))
            if ub is not None:
                output_file.write('%s.up = %s;\n' % (var_name, ftoa(ub, False)))
        else:
            raise KeyError('Category %s not supported' % category)
        if warmstart and var.value is not None:
            output_file.write('%s.l = %s;\n' % (var_name, ftoa(var.value, False)))
    if warn_int_bounds:
        logger.warning('GAMS requires finite bounds for integer variables. 1.0E100 is as extreme as GAMS will define, and should be enough to appear unbounded. If the solver cannot handle this bound, explicitly set a smaller bound on the pyomo model, or try a different GAMS solver.')
    model_name = 'GAMS_MODEL'
    output_file.write('\nMODEL %s /all/ ;\n' % model_name)
    if mtype is None:
        mtype = ('lp', 'nlp', 'mip', 'minlp')[(0 if linear else 1) + (2 if categorized_vars.binary or categorized_vars.ints else 0)]
        if mtype == 'nlp' and dnlp:
            mtype = 'dnlp'
    if solver is not None:
        if mtype.upper() not in valid_solvers[solver.upper()]:
            raise ValueError('GAMS writer passed solver (%s) unsuitable for model type (%s)' % (solver, mtype))
        output_file.write('option %s=%s;\n' % (mtype, solver))
    output_file.write('option solprint=%s;\n' % solprint)
    output_file.write('option limrow=%d;\n' % limrow)
    output_file.write('option limcol=%d;\n' % limcol)
    output_file.write('option solvelink=%d;\n' % solvelink)
    if put_results is not None and put_results_format == 'gdx':
        output_file.write('option savepoint=1;\n')
    if add_options is not None:
        output_file.write('\n* START USER ADDITIONAL OPTIONS\n')
        for line in add_options:
            output_file.write('\n' + line)
        output_file.write('\n\n* END USER ADDITIONAL OPTIONS\n\n')
    output_file.write('SOLVE %s USING %s %simizing GAMS_OBJECTIVE;\n\n' % (model_name, mtype, 'min' if obj.sense == minimize else 'max'))
    stat_vars = ['MODELSTAT', 'SOLVESTAT', 'OBJEST', 'OBJVAL', 'NUMVAR', 'NUMEQU', 'NUMDVAR', 'NUMNZ', 'ETSOLVE']
    output_file.write("Scalars MODELSTAT 'model status', SOLVESTAT 'solve status';\n")
    output_file.write('MODELSTAT = %s.modelstat;\n' % model_name)
    output_file.write('SOLVESTAT = %s.solvestat;\n\n' % model_name)
    output_file.write("Scalar OBJEST 'best objective', OBJVAL 'objective value';\n")
    output_file.write('OBJEST = %s.objest;\n' % model_name)
    output_file.write('OBJVAL = %s.objval;\n\n' % model_name)
    output_file.write("Scalar NUMVAR 'number of variables';\n")
    output_file.write('NUMVAR = %s.numvar\n\n' % model_name)
    output_file.write("Scalar NUMEQU 'number of equations';\n")
    output_file.write('NUMEQU = %s.numequ\n\n' % model_name)
    output_file.write("Scalar NUMDVAR 'number of discrete variables';\n")
    output_file.write('NUMDVAR = %s.numdvar\n\n' % model_name)
    output_file.write("Scalar NUMNZ 'number of nonzeros';\n")
    output_file.write('NUMNZ = %s.numnz\n\n' % model_name)
    output_file.write("Scalar ETSOLVE 'time to execute solve statement';\n")
    output_file.write('ETSOLVE = %s.etsolve\n\n' % model_name)
    if put_results is not None:
        if put_results_format == 'gdx':
            output_file.write("\nexecute_unload '%s_s.gdx'" % put_results)
            for stat in stat_vars:
                output_file.write(', %s' % stat)
            output_file.write(';\n')
        else:
            results = put_results + '.dat'
            output_file.write("\nfile results /'%s'/;" % results)
            output_file.write('\nresults.nd=15;')
            output_file.write('\nresults.nw=21;')
            output_file.write('\nput results;')
            output_file.write("\nput 'SYMBOL  :  LEVEL  :  MARGINAL' /;")
            for var in var_list:
                output_file.write("\nput %s ' ' %s.l ' ' %s.m /;" % (var, var, var))
            for con in constraint_names:
                output_file.write("\nput %s ' ' %s.l ' ' %s.m /;" % (con, con, con))
            output_file.write("\nput GAMS_OBJECTIVE ' ' GAMS_OBJECTIVE.l ' ' GAMS_OBJECTIVE.m;\n")
            statresults = put_results + 'stat.dat'
            output_file.write("\nfile statresults /'%s'/;" % statresults)
            output_file.write('\nstatresults.nd=15;')
            output_file.write('\nstatresults.nw=21;')
            output_file.write('\nput statresults;')
            output_file.write("\nput 'SYMBOL   :   VALUE' /;")
            for stat in stat_vars:
                output_file.write("\nput '%s' ' ' %s /;\n" % (stat, stat))