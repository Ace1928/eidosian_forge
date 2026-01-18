import ctypes
import logging
import os
from collections import deque, defaultdict, namedtuple
from contextlib import nullcontext
from itertools import filterfalse, product
from math import log10 as _log10
from operator import itemgetter, attrgetter, setitem
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.config import (
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.errors import DeveloperError, InfeasibleConstraintException, MouseTrap
from pyomo.common.gc_manager import PauseGC
from pyomo.common.numeric_types import (
from pyomo.common.timing import TicTocTimer
from pyomo.core.expr import (
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor, _EvaluationVisitor
from pyomo.core.base import (
from pyomo.core.base.component import ActiveComponent
from pyomo.core.base.constraint import _ConstraintData
from pyomo.core.base.expression import ScalarExpression, _GeneralExpressionData
from pyomo.core.base.objective import (
from pyomo.core.base.suffix import SuffixFinder
from pyomo.core.base.var import _VarData
import pyomo.core.kernel as kernel
from pyomo.core.pyomoobject import PyomoObject
from pyomo.opt import WriterFactory
from pyomo.repn.util import (
from pyomo.repn.plugins.ampl.ampl_ import set_pyomo_amplfunc_env
from pyomo.core.base import Set, RangeSet
from pyomo.network import Port
class _NLWriter_impl(object):

    def __init__(self, ostream, rowstream, colstream, config):
        self.ostream = ostream
        self.rowstream = rowstream
        self.colstream = colstream
        self.config = config
        self.symbolic_solver_labels = config.symbolic_solver_labels
        if self.symbolic_solver_labels:
            self.template = text_nl_debug_template
        else:
            self.template = text_nl_template
        self.subexpression_cache = {}
        self.subexpression_order = []
        self.external_functions = {}
        self.used_named_expressions = set()
        self.var_map = {}
        self.sorter = FileDeterminism_to_SortComponents(config.file_determinism)
        self.visitor = AMPLRepnVisitor(self.template, self.subexpression_cache, self.subexpression_order, self.external_functions, self.var_map, self.used_named_expressions, self.symbolic_solver_labels, self.config.export_defined_variables, self.sorter)
        self.next_V_line_id = 0
        self.pause_gc = None

    def __enter__(self):
        assert AMPLRepn.ActiveVisitor is None
        AMPLRepn.ActiveVisitor = self.visitor
        self.pause_gc = PauseGC()
        self.pause_gc.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.pause_gc.__exit__(exc_type, exc_value, tb)
        assert AMPLRepn.ActiveVisitor is self.visitor
        AMPLRepn.ActiveVisitor = None

    def write(self, model):
        timing_logger = logging.getLogger('pyomo.common.timing.writer')
        timer = TicTocTimer(logger=timing_logger)
        with_debug_timing = timing_logger.isEnabledFor(logging.DEBUG) and timing_logger.hasHandlers()
        sorter = FileDeterminism_to_SortComponents(self.config.file_determinism)
        component_map, unknown = categorize_valid_components(model, active=True, sort=sorter, valid={Block, Objective, Constraint, Var, Param, Expression, ExternalFunction, Set, RangeSet, Port}, targets={Suffix, SOSConstraint})
        if unknown:
            raise ValueError("The model ('%s') contains the following active components that the NL writer does not know how to process:\n\t%s" % (model.name, '\n\t'.join(('%s:\n\t\t%s' % (k, '\n\t\t'.join(map(attrgetter('name'), v))) for k, v in unknown.items()))))
        symbolic_solver_labels = self.symbolic_solver_labels
        visitor = self.visitor
        ostream = self.ostream
        linear_presolve = self.config.linear_presolve
        var_map = self.var_map
        initialize_var_map_from_column_order(model, self.config, var_map)
        timer.toc('Initialized column order', level=logging.DEBUG)
        suffix_data = {}
        if component_map[Suffix]:
            for block in reversed(component_map[Suffix]):
                for suffix in block.component_objects(Suffix, active=True, descend_into=False, sort=sorter):
                    if not suffix.export_enabled() or not suffix:
                        continue
                    name = suffix.local_name
                    if name not in suffix_data:
                        suffix_data[name] = _SuffixData(name)
                    suffix_data[name].update(suffix)
        if self.config.scale_model and 'scaling_factor' in suffix_data:
            scaling_factor = CachingNumericSuffixFinder('scaling_factor', 1)
            scaling_cache = scaling_factor.suffix_cache
            del suffix_data['scaling_factor']
        else:
            scaling_factor = _NoScalingFactor()
        scale_model = scaling_factor.scale
        timer.toc('Collected suffixes', level=logging.DEBUG)
        lcon_by_linear_nnz = defaultdict(dict)
        comp_by_linear_var = defaultdict(list)
        objectives = []
        linear_objs = []
        last_parent = None
        for obj in model.component_data_objects(Objective, active=True, sort=sorter):
            if with_debug_timing and obj.parent_component() is not last_parent:
                if last_parent is None:
                    timer.toc(None)
                else:
                    timer.toc('Objective %s', last_parent, level=logging.DEBUG)
                last_parent = obj.parent_component()
            expr_info = visitor.walk_expression((obj.expr, obj, 1, scaling_factor(obj)))
            if expr_info.named_exprs:
                self._record_named_expression_usage(expr_info.named_exprs, obj, 1)
            if expr_info.nonlinear:
                objectives.append((obj, expr_info))
            else:
                linear_objs.append((obj, expr_info))
            if linear_presolve:
                obj_id = id(obj)
                for _id in expr_info.linear:
                    comp_by_linear_var[_id].append((obj_id, expr_info))
        if with_debug_timing:
            timer.toc('Objective %s', last_parent, level=logging.DEBUG)
        else:
            timer.toc('Processed %s objectives', len(objectives))
        n_nonlinear_objs = len(objectives)
        objectives.extend(linear_objs)
        n_objs = len(objectives)
        constraints = []
        linear_cons = []
        n_ranges = 0
        n_equality = 0
        n_complementarity_nonlin = 0
        n_complementarity_lin = 0
        n_complementarity_range = 0
        n_complementarity_nz_var_lb = 0
        last_parent = None
        for con in ordered_active_constraints(model, self.config):
            if with_debug_timing and con.parent_component() is not last_parent:
                if last_parent is None:
                    timer.toc(None)
                else:
                    timer.toc('Constraint %s', last_parent, level=logging.DEBUG)
                last_parent = con.parent_component()
            scale = scaling_factor(con)
            expr_info = visitor.walk_expression((con.body, con, 0, scale))
            if expr_info.named_exprs:
                self._record_named_expression_usage(expr_info.named_exprs, con, 0)
            lb = con.lb
            ub = con.ub
            if lb is None and ub is None:
                continue
            if scale != 1:
                if lb is not None:
                    lb = lb * scale
                if ub is not None:
                    ub = ub * scale
                if scale < 0:
                    lb, ub = (ub, lb)
            if expr_info.nonlinear:
                constraints.append((con, expr_info, lb, ub))
            elif expr_info.linear:
                linear_cons.append((con, expr_info, lb, ub))
            elif not self.config.skip_trivial_constraints:
                linear_cons.append((con, expr_info, lb, ub))
            else:
                c = expr_info.const
                if lb is not None and lb - c > TOL or (ub is not None and ub - c < -TOL):
                    raise InfeasibleConstraintException(f"model contains a trivially infeasible constraint '{con.name}' (fixed body value {c} outside bounds [{lb}, {ub}]).")
            if linear_presolve:
                con_id = id(con)
                if not expr_info.nonlinear and lb == ub and (lb is not None):
                    lcon_by_linear_nnz[len(expr_info.linear)][con_id] = (expr_info, lb)
                for _id in expr_info.linear:
                    comp_by_linear_var[_id].append((con_id, expr_info))
        if with_debug_timing:
            timer.toc('Constraint %s', last_parent, level=logging.DEBUG)
        else:
            timer.toc('Processed %s constraints', len(constraints))
        var_bounds = {_id: v.bounds for _id, v in var_map.items()}
        eliminated_cons, eliminated_vars = self._linear_presolve(comp_by_linear_var, lcon_by_linear_nnz, var_bounds)
        del comp_by_linear_var
        del lcon_by_linear_nnz
        n_nonlinear_cons = len(constraints)
        if eliminated_cons:
            _removed = eliminated_cons.__contains__
            constraints.extend(filterfalse(lambda c: _removed(id(c[0])), linear_cons))
        else:
            constraints.extend(linear_cons)
        n_cons = len(constraints)
        self.subexpression_order = list(filter(self.used_named_expressions.__contains__, self.subexpression_order))
        linear_by_comp = {_id: info.linear for _id, info in eliminated_vars.items()}
        self._categorize_vars(map(self.subexpression_cache.__getitem__, self.subexpression_order), linear_by_comp)
        n_subexpressions = self._count_subexpression_occurrences()
        obj_vars_linear, obj_vars_nonlinear, obj_nnz_by_var = self._categorize_vars(objectives, linear_by_comp)
        con_vars_linear, con_vars_nonlinear, con_nnz_by_var = self._categorize_vars(constraints, linear_by_comp)
        if self.config.export_nonlinear_variables:
            for v in self.config.export_nonlinear_variables:
                if v.is_indexed():
                    _iter = v.values(sorter)
                else:
                    _iter = (v,)
                for _v in _iter:
                    _id = id(_v)
                    if _id not in var_map:
                        var_map[_id] = _v
                        var_bounds[_id] = _v.bounds
                    con_vars_nonlinear.add(_id)
        con_nnz = sum(con_nnz_by_var.values())
        timer.toc('Categorized model variables: %s nnz', con_nnz, level=logging.DEBUG)
        n_lcons = 0
        for block in component_map[SOSConstraint]:
            for sos in block.component_data_objects(SOSConstraint, active=True, descend_into=False, sort=sorter):
                for v in sos.variables:
                    if id(v) not in var_map:
                        _id = id(v)
                        var_map[_id] = v
                        con_vars_linear.add(_id)
        obj_vars = obj_vars_linear | obj_vars_nonlinear
        con_vars = con_vars_linear | con_vars_nonlinear
        all_vars = con_vars | obj_vars
        n_vars = len(all_vars)
        continuous_vars = set()
        binary_vars = set()
        integer_vars = set()
        for _id in all_vars:
            v = var_map[_id]
            if v.is_continuous():
                continuous_vars.add(_id)
            elif v.is_binary():
                binary_vars.add(_id)
            elif v.is_integer():
                integer_vars.add(_id)
            else:
                raise ValueError(f"Variable '{v.name}' has a domain that is not Real, Integer, or Binary: Cannot write a legal NL file.")
        discrete_vars = binary_vars | integer_vars
        nonlinear_vars = con_vars_nonlinear | obj_vars_nonlinear
        linear_only_vars = (con_vars_linear | obj_vars_linear) - nonlinear_vars
        self.column_order = column_order = {_id: i for i, _id in enumerate(var_map)}
        variables = []
        both_vars_nonlinear = con_vars_nonlinear & obj_vars_nonlinear
        if both_vars_nonlinear:
            variables.extend(sorted(both_vars_nonlinear & continuous_vars, key=column_order.__getitem__))
            variables.extend(sorted(both_vars_nonlinear & discrete_vars, key=column_order.__getitem__))
        con_only_nonlinear_vars = con_vars_nonlinear - both_vars_nonlinear
        if con_only_nonlinear_vars:
            variables.extend(sorted(con_only_nonlinear_vars & continuous_vars, key=column_order.__getitem__))
            variables.extend(sorted(con_only_nonlinear_vars & discrete_vars, key=column_order.__getitem__))
        obj_only_nonlinear_vars = obj_vars_nonlinear - both_vars_nonlinear
        if obj_vars_nonlinear:
            variables.extend(sorted(obj_only_nonlinear_vars & continuous_vars, key=column_order.__getitem__))
            variables.extend(sorted(obj_only_nonlinear_vars & discrete_vars, key=column_order.__getitem__))
        if linear_only_vars:
            variables.extend(sorted(linear_only_vars - discrete_vars, key=column_order.__getitem__))
            linear_binary_vars = linear_only_vars & binary_vars
            variables.extend(sorted(linear_binary_vars, key=column_order.__getitem__))
            linear_integer_vars = linear_only_vars & integer_vars
            variables.extend(sorted(linear_integer_vars, key=column_order.__getitem__))
        else:
            linear_binary_vars = linear_integer_vars = set()
        assert len(variables) == n_vars
        timer.toc('Set row / column ordering: %s var [%s, %s, %s R/B/Z], %s con [%s, %s L/NL]', n_vars, len(continuous_vars), len(binary_vars), len(integer_vars), len(constraints), n_cons - n_nonlinear_cons, n_nonlinear_cons, level=logging.DEBUG)
        self.column_order = column_order = {_id: i for i, _id in enumerate(variables)}
        if component_map[SOSConstraint]:
            for name in ('sosno', 'ref'):
                if name in suffix_data:
                    raise RuntimeError(f"The Pyomo NL file writer does not allow both manually declared '{name}' suffixes as well as SOSConstraint components to exist on a single model. To avoid this error please use only one of these methods to define special ordered sets.")
                suffix_data[name] = _SuffixData(name)
                suffix_data[name].datatype.add(Suffix.INT)
            sos_id = 0
            sosno = suffix_data['sosno']
            ref = suffix_data['ref']
            for block in reversed(component_map[SOSConstraint]):
                for sos in block.component_data_objects(SOSConstraint, active=True, descend_into=False, sort=sorter):
                    sos_id += 1
                    if sos.level == 1:
                        tag = sos_id
                    elif sos.level == 2:
                        tag = -sos_id
                    else:
                        raise ValueError(f"SOSConstraint '{sos.name}' has sos type='{sos.level}', which is not supported by the NL file interface")
                    try:
                        _items = sos.get_items()
                    except AttributeError:
                        _items = sos.items()
                    for v, r in _items:
                        sosno.store(v, tag)
                        ref.store(v, r)
        if suffix_data:
            row_order = {id(con[0]): i for i, con in enumerate(constraints)}
            obj_order = {id(obj[0]): i for i, obj in enumerate(objectives)}
            model_id = id(model)
        if symbolic_solver_labels:
            labeler = NameLabeler()
            row_labels = [labeler(info[0]) for info in constraints] + [labeler(info[0]) for info in objectives]
            row_comments = [f'\t#{lbl}' for lbl in row_labels]
            col_labels = [labeler(var_map[_id]) for _id in variables]
            col_comments = [f'\t#{lbl}' for lbl in col_labels]
            self.var_id_to_nl = {_id: f'v{var_idx}{col_comments[var_idx]}' for var_idx, _id in enumerate(variables)}
            if self.rowstream is not None:
                self.rowstream.write('\n'.join(row_labels))
                self.rowstream.write('\n')
            if self.colstream is not None:
                self.colstream.write('\n'.join(col_labels))
                self.colstream.write('\n')
        else:
            row_labels = row_comments = [''] * (n_cons + n_objs)
            col_labels = col_comments = [''] * len(variables)
            self.var_id_to_nl = {_id: f'v{var_idx}' for var_idx, _id in enumerate(variables)}
        _vmap = self.var_id_to_nl
        if scale_model:
            template = self.template
            objective_scaling = [scaling_cache[id(info[0])] for info in objectives]
            constraint_scaling = [scaling_cache[id(info[0])] for info in constraints]
            variable_scaling = [scaling_factor(var_map[_id]) for _id in variables]
            for _id, scale in zip(variables, variable_scaling):
                if scale == 1:
                    continue
                if scale < 0:
                    ub, lb = var_bounds[_id]
                else:
                    lb, ub = var_bounds[_id]
                if lb is not None:
                    lb *= scale
                if ub is not None:
                    ub *= scale
                var_bounds[_id] = (lb, ub)
                _vmap[_id] = (template.division + _vmap[_id] + '\n' + template.const % scale).rstrip()
        for _id, expr_info in eliminated_vars.items():
            nl, args, _ = expr_info.compile_repn(visitor)
            _vmap[_id] = nl.rstrip() % tuple((_vmap[_id] for _id in args))
        r_lines = [None] * n_cons
        for idx, (con, expr_info, lb, ub) in enumerate(constraints):
            if lb == ub:
                if lb is None:
                    r_lines[idx] = '3'
                else:
                    r_lines[idx] = f'4 {lb - expr_info.const!r}'
                    n_equality += 1
            elif lb is None:
                r_lines[idx] = f'1 {ub - expr_info.const!r}'
            elif ub is None:
                r_lines[idx] = f'2 {lb - expr_info.const!r}'
            else:
                r_lines[idx] = f'0 {lb - expr_info.const!r} {ub - expr_info.const!r}'
                n_ranges += 1
            expr_info.const = 0
            if hasattr(con, '_complementarity'):
                r_lines[idx] = f'5 {con._complementarity} {1 + column_order[con._vid]}'
                if expr_info.nonlinear:
                    n_complementarity_nonlin += 1
                else:
                    n_complementarity_lin += 1
        if symbolic_solver_labels:
            for idx in range(len(constraints)):
                r_lines[idx] += row_comments[idx]
        timer.toc('Generated row/col labels & comments', level=logging.DEBUG)
        if visitor.encountered_string_arguments and 'b' not in getattr(ostream, 'mode', ''):
            try:
                _written_bytes = ostream.tell()
            except IOError:
                _written_bytes = None
        line_1_txt = f'g3 1 1 0\t# problem {model.name}\n'
        ostream.write(line_1_txt)
        if visitor.encountered_string_arguments and 'b' not in getattr(ostream, 'mode', ''):
            if _written_bytes is None:
                _written_bytes = 0
            else:
                _written_bytes = ostream.tell() - _written_bytes
            if not _written_bytes:
                if os.linesep != '\n':
                    logger.warning("Writing NL file containing string arguments to a text output stream that does not support tell() on a platform with default line endings other than '\\n'. Current versions of the ASL (through at least 20190605) require UNIX-style newlines as terminators for string arguments: it is possible that the ASL may refuse to read the NL file.")
            else:
                if ostream.encoding:
                    line_1_txt = line_1_txt.encode(ostream.encoding)
                if len(line_1_txt) != _written_bytes:
                    logger.error("Writing NL file containing string arguments to a text output stream with line endings other than '\\n' Current versions of the ASL (through at least 20190605) require UNIX-style newlines as terminators for string arguments.")
        ostream.write(' %d %d %d %d %d \t# vars, constraints, objectives, ranges, eqns\n' % (n_vars, n_cons, n_objs, n_ranges, n_equality))
        ostream.write(' %d %d %d %d %d %d\t# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb\n' % (n_nonlinear_cons, n_nonlinear_objs, n_complementarity_lin, n_complementarity_nonlin, n_complementarity_range, n_complementarity_nz_var_lb))
        ostream.write(' 0 0\t# network constraints: nonlinear, linear\n')
        _n_both_vars = len(both_vars_nonlinear)
        _n_con_vars = len(con_vars_nonlinear)
        _n_obj_vars = _n_con_vars + len(obj_vars_nonlinear) - _n_both_vars
        if _n_obj_vars == _n_con_vars:
            _n_obj_vars = _n_both_vars
        ostream.write(' %d %d %d \t# nonlinear vars in constraints, objectives, both\n' % (_n_con_vars, _n_obj_vars, _n_both_vars))
        ostream.write(' 0 %d 0 1\t# linear network variables; functions; arith, flags\n' % (len(self.external_functions),))
        ostream.write(' %d %d %d %d %d \t# discrete variables: binary, integer, nonlinear (b,c,o)\n' % (len(linear_binary_vars), len(linear_integer_vars), len(both_vars_nonlinear.intersection(discrete_vars)), len(con_vars_nonlinear.intersection(discrete_vars)), len(obj_vars_nonlinear.intersection(discrete_vars))))
        ostream.write(' %d %d \t# nonzeros in Jacobian, obj. gradient\n' % (sum(con_nnz_by_var.values()), sum(obj_nnz_by_var.values())))
        ostream.write(' %d %d\t# max name lengths: constraints, variables\n' % (max(map(len, row_labels), default=0), max(map(len, col_labels), default=0)))
        ostream.write(' %d %d %d %d %d\t# common exprs: b,c,o,c1,o1\n' % tuple(n_subexpressions))
        amplfunc_libraries = set()
        for fid, fcn in sorted(self.external_functions.values()):
            amplfunc_libraries.add(fcn._library)
            ostream.write('F%d 1 -1 %s\n' % (fid, fcn._function))
        for name, data in suffix_data.items():
            if name == 'dual':
                continue
            data.compile(column_order, row_order, obj_order, model_id)
            if len(data.datatype) > 1:
                raise ValueError("The NL file writer found multiple active export suffix components with name '{name}' and different datatypes. A single datatype must be declared.")
            _type = next(iter(data.datatype))
            if _type == Suffix.FLOAT:
                _float = 4
            elif _type == Suffix.INT:
                _float = 0
            else:
                raise ValueError(f"The NL file writer only supports export suffixes declared with a numeric datatype.  Suffix component '{name}' declares type '{_type}'")
            for _field, _vals in zip(range(4), (data.var, data.con, data.obj, data.prob)):
                if not _vals:
                    continue
                ostream.write(f'S{_field | _float} {len(_vals)} {name}\n')
                ostream.write(''.join((f'{_id} {_vals[_id]!r}\n' for _id in sorted(_vals))))
        single_use_subexpressions = {}
        self.next_V_line_id = n_vars
        for _id in self.subexpression_order:
            _con_id, _obj_id, _sub = self.subexpression_cache[_id][2]
            if _sub:
                continue
            target_expr = 0
            if _obj_id is None:
                target_expr = _con_id
            elif _con_id is None:
                target_expr = _obj_id
            if target_expr == 0:
                self._write_v_line(_id, 0)
            else:
                if target_expr not in single_use_subexpressions:
                    single_use_subexpressions[target_expr] = []
                single_use_subexpressions[target_expr].append(_id)
        for row_idx, info in enumerate(constraints):
            if info[1].nonlinear is None:
                _expr = self.template.const % 0
                if symbolic_solver_labels:
                    ostream.write(_expr.join((f'C{i}{row_comments[i]}\n' for i in range(row_idx, len(constraints)))))
                else:
                    ostream.write(_expr.join((f'C{i}\n' for i in range(row_idx, len(constraints)))))
                ostream.write(_expr)
                break
            if single_use_subexpressions:
                for _id in single_use_subexpressions.get(id(info[0]), ()):
                    self._write_v_line(_id, row_idx + 1)
            ostream.write(f'C{row_idx}{row_comments[row_idx]}\n')
            self._write_nl_expression(info[1], False)
        for obj_idx, info in enumerate(objectives):
            if single_use_subexpressions:
                for _id in single_use_subexpressions.get(id(info[0]), ()):
                    self._write_v_line(_id, n_cons + n_lcons + obj_idx + 1)
            lbl = row_comments[n_cons + obj_idx]
            sense = 0 if info[0].sense == minimize else 1
            ostream.write(f'O{obj_idx} {sense}{lbl}\n')
            self._write_nl_expression(info[1], True)
        if 'dual' in suffix_data:
            data = suffix_data['dual']
            data.compile(column_order, row_order, obj_order, model_id)
            if scale_model:
                if objectives:
                    if len(objectives) > 1:
                        logger.warning('Scaling model with dual suffixes and multiple objectives.  Assuming that the duals are computed against the first objective.')
                    _obj_scale = objective_scaling[0]
                else:
                    _obj_scale = 1
                for i in data.con:
                    data.con[i] *= _obj_scale / constraint_scaling[i]
            if data.var:
                logger.warning("ignoring 'dual' suffix for Var types")
            if data.obj:
                logger.warning("ignoring 'dual' suffix for Objective types")
            if data.prob:
                logger.warning("ignoring 'dual' suffix for Model")
            if data.con:
                ostream.write(f'd{len(data.con)}\n')
                ostream.write(''.join((f'{_id} {data.con[_id]!r}\n' for _id in sorted(data.con))))
        _init_lines = [(var_idx, val if val.__class__ in int_float else float(val)) for var_idx, val in enumerate((var_map[_id].value for _id in variables)) if val is not None]
        if scale_model:
            _init_lines = [(var_idx, val * variable_scaling[var_idx]) for var_idx, val in _init_lines]
        ostream.write('x%d%s\n' % (len(_init_lines), '\t# initial guess' if symbolic_solver_labels else ''))
        ostream.write(''.join((f'{var_idx} {val!r}{col_comments[var_idx]}\n' for var_idx, val in _init_lines)))
        ostream.write('r%s\n' % ("\t#%d ranges (rhs's)" % len(constraints) if symbolic_solver_labels else '',))
        ostream.write('\n'.join(r_lines))
        if r_lines:
            ostream.write('\n')
        ostream.write('b%s\n' % ('\t#%d bounds (on variables)' % len(variables) if symbolic_solver_labels else '',))
        for var_idx, _id in enumerate(variables):
            lb, ub = var_bounds[_id]
            if lb == ub:
                if lb is None:
                    ostream.write(f'3{col_comments[var_idx]}\n')
                else:
                    ostream.write(f'4 {lb!r}{col_comments[var_idx]}\n')
            elif lb is None:
                ostream.write(f'1 {ub!r}{col_comments[var_idx]}\n')
            elif ub is None:
                ostream.write(f'2 {lb!r}{col_comments[var_idx]}\n')
            else:
                ostream.write(f'0 {lb!r} {ub!r}{col_comments[var_idx]}\n')
        ostream.write('k%d%s\n' % (len(variables) - 1, '\t#intermediate Jacobian column lengths' if symbolic_solver_labels else ''))
        ktot = 0
        for var_idx, _id in enumerate(variables[:-1]):
            ktot += con_nnz_by_var.get(_id, 0)
            ostream.write(f'{ktot}\n')
        for row_idx, info in enumerate(constraints):
            linear = info[1].linear
            if not linear:
                continue
            if scale_model:
                for _id, val in linear.items():
                    linear[_id] /= scaling_cache[_id]
            ostream.write(f'J{row_idx} {len(linear)}{row_comments[row_idx]}\n')
            for _id in sorted(linear, key=column_order.__getitem__):
                ostream.write(f'{column_order[_id]} {linear[_id]!r}\n')
        for obj_idx, info in enumerate(objectives):
            linear = info[1].linear
            if not linear:
                continue
            if scale_model:
                for _id, val in linear.items():
                    linear[_id] /= scaling_cache[_id]
            ostream.write(f'G{obj_idx} {len(linear)}{row_comments[obj_idx + n_cons]}\n')
            for _id in sorted(linear, key=column_order.__getitem__):
                ostream.write(f'{column_order[_id]} {linear[_id]!r}\n')
        eliminated_vars = [(var_map[_id], expr_info.to_expr(var_map)) for _id, expr_info in eliminated_vars.items()]
        eliminated_vars.reverse()
        if scale_model:
            scaling = ScalingFactors(variables=variable_scaling, constraints=constraint_scaling, objectives=objective_scaling)
        else:
            scaling = None
        info = NLWriterInfo(var=[var_map[_id] for _id in variables], con=[info[0] for info in constraints], obj=[info[0] for info in objectives], external_libs=sorted(amplfunc_libraries), row_labels=row_labels, col_labels=col_labels, eliminated_vars=eliminated_vars, scaling=scaling)
        timer.toc('Wrote NL stream', level=logging.DEBUG)
        timer.toc('Generated NL representation', delta=False)
        return info

    def _categorize_vars(self, comp_list, linear_by_comp):
        """Categorize compiled expression vars into linear and nonlinear

        This routine takes an iterable of compiled component expression
        infos and returns the sets of variables appearing linearly and
        nonlinearly in those components.

        This routine has a number of side effects:

          - the ``linear_by_comp`` dict is updated to contain the set of
            nonzeros for each component in the ``comp_list``

          - the expr_info (the second element in each tuple in
            ``comp_list``) is "compiled": the ``linear`` attribute is
            converted from a list of coef, var_id terms (potentially with
            duplicate entries) into a dict that maps var id to
            coefficients

        Returns
        -------
        all_linear_vars: set
            set of all vars that only appear linearly in the compiled
            component expression infos

        all_nonlinear_vars: set
            set of all vars that appear nonlinearly in the compiled
            component expression infos

        nnz_by_var: dict
            Count of the number of components that each var appears in.

        """
        all_linear_vars = set()
        all_nonlinear_vars = set()
        nnz_by_var = {}
        for comp_info in comp_list:
            expr_info = comp_info[1]
            if expr_info.linear:
                linear_vars = set(expr_info.linear)
                all_linear_vars.update(linear_vars)
            if expr_info.nonlinear:
                nonlinear_vars = set()
                for _id in expr_info.nonlinear[1]:
                    if _id in nonlinear_vars:
                        continue
                    if _id in linear_by_comp:
                        nonlinear_vars.update(linear_by_comp[_id])
                    else:
                        nonlinear_vars.add(_id)
                if expr_info.linear:
                    for i in filterfalse(linear_vars.__contains__, nonlinear_vars):
                        expr_info.linear[i] = 0
                else:
                    expr_info.linear = dict.fromkeys(nonlinear_vars, 0)
                all_nonlinear_vars.update(nonlinear_vars)
            for v in expr_info.linear:
                if v in nnz_by_var:
                    nnz_by_var[v] += 1
                else:
                    nnz_by_var[v] = 1
            linear_by_comp[id(comp_info[0])] = expr_info.linear
        if all_nonlinear_vars:
            all_linear_vars -= all_nonlinear_vars
        return (all_linear_vars, all_nonlinear_vars, nnz_by_var)

    def _count_subexpression_occurrences(self):
        """Categorize named subexpressions based on where they are used.

        This iterates through the `subexpression_order` and categorizes
        each _id based on where it is used (1 constraint, many
        constraints, 1 objective, many objectives, constraints and
        objectives).

        """
        n_subexpressions = [0] * 5
        for info in map(itemgetter(2), map(self.subexpression_cache.__getitem__, self.subexpression_order)):
            if info[2]:
                pass
            elif info[1] is None:
                n_subexpressions[3 if info[0] else 1] += 1
            elif info[0] is None:
                n_subexpressions[4 if info[1] else 2] += 1
            else:
                n_subexpressions[0] += 1
        return n_subexpressions

    def _linear_presolve(self, comp_by_linear_var, lcon_by_linear_nnz, var_bounds):
        eliminated_vars = {}
        eliminated_cons = set()
        if not self.config.linear_presolve:
            return (eliminated_cons, eliminated_vars)
        for expr, info, _ in self.subexpression_cache.values():
            if not info.linear:
                continue
            expr_id = id(expr)
            for _id in info.linear:
                comp_by_linear_var[_id].append((expr_id, info))
        fixed_vars = [_id for _id, (lb, ub) in var_bounds.items() if lb == ub and lb is not None]
        var_map = self.var_map
        substitutions_by_linear_var = defaultdict(set)
        template = self.template
        one_var = lcon_by_linear_nnz[1]
        two_var = lcon_by_linear_nnz[2]
        while 1:
            if fixed_vars:
                _id = fixed_vars.pop()
                a = x = None
                b, _ = var_bounds[_id]
                logger.debug('NL presolve: bounds fixed %s := %s', var_map[_id], b)
                eliminated_vars[_id] = AMPLRepn(b, {}, None)
            elif one_var:
                con_id, info = one_var.popitem()
                expr_info, lb = info
                _id, coef = expr_info.linear.popitem()
                a = x = None
                b = expr_info.const = (lb - expr_info.const) / coef
                logger.debug('NL presolve: substituting %s := %s', var_map[_id], b)
                eliminated_vars[_id] = expr_info
                lb, ub = var_bounds[_id]
                if lb is not None and lb - b > TOL or (ub is not None and ub - b < -TOL):
                    raise InfeasibleConstraintException(f"model contains a trivially infeasible variable '{var_map[_id].name}' (presolved to a value of {b} outside bounds [{lb}, {ub}]).")
                eliminated_cons.add(con_id)
            elif two_var:
                con_id, info = two_var.popitem()
                expr_info, lb = info
                _id, coef = expr_info.linear.popitem()
                id2, coef2 = expr_info.linear.popitem()
                id2_isdiscrete = var_map[id2].domain.isdiscrete()
                if var_map[_id].domain.isdiscrete() ^ id2_isdiscrete:
                    if id2_isdiscrete:
                        _id, id2 = (id2, _id)
                        coef, coef2 = (coef2, coef)
                else:
                    log_coef = _log10(abs(coef))
                    log_coef2 = _log10(abs(coef2))
                    if abs(log_coef2) < abs(log_coef) or (log_coef2 == -log_coef and log_coef2 < log_coef):
                        _id, id2 = (id2, _id)
                        coef, coef2 = (coef2, coef)
                a = -coef2 / coef
                x = id2
                b = expr_info.const = (lb - expr_info.const) / coef
                expr_info.linear[x] = a
                substitutions_by_linear_var[x].add(_id)
                eliminated_vars[_id] = expr_info
                logger.debug('NL presolve: substituting %s := %s*%s + %s', var_map[_id], a, var_map[x], b)
                x_lb, x_ub = var_bounds[x]
                lb, ub = var_bounds[_id]
                if lb is not None:
                    lb = (lb - b) / a
                if ub is not None:
                    ub = (ub - b) / a
                if a < 0:
                    lb, ub = (ub, lb)
                if x_lb is None or (lb is not None and lb > x_lb):
                    x_lb = lb
                if x_ub is None or (ub is not None and ub < x_ub):
                    x_ub = ub
                var_bounds[x] = (x_lb, x_ub)
                if x_lb == x_ub and x_lb is not None:
                    fixed_vars.append(x)
                eliminated_cons.add(con_id)
            else:
                return (eliminated_cons, eliminated_vars)
            for con_id, expr_info in comp_by_linear_var[_id]:
                c = expr_info.linear.pop(_id, 0)
                expr_info.const += c * b
                if x in expr_info.linear:
                    expr_info.linear[x] += c * a
                elif a:
                    expr_info.linear[x] = c * a
                    comp_by_linear_var[x].append((con_id, expr_info))
                    continue
                nnz = len(expr_info.linear)
                _old = lcon_by_linear_nnz[nnz + 1]
                if con_id in _old:
                    lcon_by_linear_nnz[nnz][con_id] = _old.pop(con_id)
            for resubst in substitutions_by_linear_var.pop(_id, ()):
                expr_info = eliminated_vars[resubst]
                c = expr_info.linear.pop(_id, 0)
                expr_info.const += c * b
                if x in expr_info.linear:
                    expr_info.linear[x] += c * a
                elif a:
                    expr_info.linear[x] = c * a

    def _record_named_expression_usage(self, named_exprs, src, comp_type):
        self.used_named_expressions.update(named_exprs)
        src = id(src)
        for _id in named_exprs:
            info = self.subexpression_cache[_id][2]
            if info[comp_type] is None:
                info[comp_type] = src
            elif info[comp_type] != src:
                info[comp_type] = 0

    def _write_nl_expression(self, repn, include_const):
        if repn.nonlinear:
            nl, args = repn.nonlinear
            if include_const and repn.const:
                nl = self.template.binary_sum + nl + self.template.const % repn.const
            self.ostream.write(nl % tuple(map(self.var_id_to_nl.__getitem__, args)))
        elif include_const:
            self.ostream.write(self.template.const % repn.const)
        else:
            self.ostream.write(self.template.const % 0)

    def _write_v_line(self, expr_id, k):
        ostream = self.ostream
        column_order = self.column_order
        info = self.subexpression_cache[expr_id]
        if self.symbolic_solver_labels:
            lbl = '\t#%s' % info[0].name
        else:
            lbl = ''
        self.var_id_to_nl[expr_id] = f'v{self.next_V_line_id}{lbl}'
        linear = dict((item for item in info[1].linear.items() if item[1]))
        ostream.write(f'V{self.next_V_line_id} {len(linear)} {k}{lbl}\n')
        for _id in sorted(linear, key=column_order.__getitem__):
            ostream.write(f'{column_order[_id]} {linear[_id]!r}\n')
        self._write_nl_expression(info[1], True)
        self.next_V_line_id += 1