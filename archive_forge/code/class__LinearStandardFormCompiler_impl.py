import collections
import logging
from operator import attrgetter
from pyomo.common.config import (
from pyomo.common.dependencies import scipy, numpy as np
from pyomo.common.gc_manager import PauseGC
from pyomo.common.timing import TicTocTimer
from pyomo.core.base import (
from pyomo.opt import WriterFactory
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.repn.util import (
from pyomo.core.base import Set, RangeSet, ExternalFunction
from pyomo.network import Port
class _LinearStandardFormCompiler_impl(object):

    def __init__(self, config):
        self.config = config

    def write(self, model):
        timing_logger = logging.getLogger('pyomo.common.timing.writer')
        timer = TicTocTimer(logger=timing_logger)
        with_debug_timing = timing_logger.isEnabledFor(logging.DEBUG) and timing_logger.hasHandlers()
        sorter = FileDeterminism_to_SortComponents(self.config.file_determinism)
        component_map, unknown = categorize_valid_components(model, active=True, sort=sorter, valid={Block, Constraint, Var, Param, Expression, ExternalFunction, Set, RangeSet, Port}, targets={Suffix, Objective})
        if unknown:
            raise ValueError("The model ('%s') contains the following active components that the Linear Standard Form compiler does not know how to process:\n\t%s" % (model.name, '\n\t'.join(('%s:\n\t\t%s' % (k, '\n\t\t'.join(map(attrgetter('name'), v))) for k, v in unknown.items()))))
        self.var_map = var_map = {}
        initialize_var_map_from_column_order(model, self.config, var_map)
        var_order = {_id: i for i, _id in enumerate(var_map)}
        visitor = LinearRepnVisitor({}, var_map, var_order, sorter)
        timer.toc('Initialized column order', level=logging.DEBUG)
        if component_map[Suffix]:
            suffixesByName = {}
            for block in component_map[Suffix]:
                for suffix in block.component_objects(Suffix, active=True, descend_into=False, sort=sorter):
                    if not suffix.export_enabled() or not suffix:
                        continue
                    name = suffix.local_name
                    if name in suffixesByName:
                        suffixesByName[name].append(suffix)
                    else:
                        suffixesByName[name] = [suffix]
            for name, suffixes in suffixesByName.items():
                n = len(suffixes)
                plural = 's' if n > 1 else ''
                logger.warning(f"EXPORT Suffix '{name}' found on {n} block{plural}:\n    " + '\n    '.join((s.name for s in suffixes)) + '\nStandard Form compiler ignores export suffixes.  Skipping.')
        if not component_map[Objective]:
            objectives = [Objective(expr=1)]
            objectives[0].construct()
        else:
            objectives = []
            for blk in component_map[Objective]:
                objectives.extend(blk.component_data_objects(Objective, active=True, descend_into=False, sort=sorter))
        obj_data = []
        obj_index = []
        obj_index_ptr = [0]
        for i, obj in enumerate(objectives):
            repn = visitor.walk_expression(obj.expr)
            if repn.nonlinear is not None:
                raise ValueError(f'Model objective ({obj.name}) contains nonlinear terms that cannot be compiled to standard (linear) form.')
            N = len(repn.linear)
            obj_data.append(np.fromiter(repn.linear.values(), float, N))
            if obj.sense == maximize:
                obj_data[-1] *= -1
            obj_index.append(np.fromiter(map(var_order.__getitem__, repn.linear), float, N))
            obj_index_ptr.append(obj_index_ptr[-1] + N)
            if with_debug_timing:
                timer.toc('Objective %s', obj, level=logging.DEBUG)
        slack_form = self.config.slack_form
        rows = []
        rhs = []
        con_data = []
        con_index = []
        con_index_ptr = [0]
        last_parent = None
        for con in ordered_active_constraints(model, self.config):
            if with_debug_timing and con.parent_component() is not last_parent:
                if last_parent is not None:
                    timer.toc('Constraint %s', last_parent, level=logging.DEBUG)
                last_parent = con.parent_component()
            lb = con.lb
            ub = con.ub
            repn = visitor.walk_expression(con.body)
            if lb is None and ub is None:
                continue
            if repn.nonlinear is not None:
                raise ValueError(f'Model constraint ({con.name}) contains nonlinear terms that cannot be compiled to standard (linear) form.')
            offset = repn.constant
            repn.constant = 0
            if not repn.linear:
                if (lb is None or lb <= offset) and (ub is None or ub >= offset):
                    continue
                raise InfeasibleError(f"model contains a trivially infeasible constraint, '{con.name}'")
            if slack_form:
                _data = list(repn.linear.values())
                _index = list(map(var_order.__getitem__, repn.linear))
                if lb == ub:
                    rhs.append(ub - offset)
                else:
                    v = Var(name=f'_slack_{len(rhs)}', bounds=(None, None))
                    v.construct()
                    if lb is None:
                        rhs.append(ub - offset)
                        v.lb = 0
                    else:
                        rhs.append(lb - offset)
                        v.ub = 0
                        if ub is not None:
                            v.lb = lb - ub
                    var_map[id(v)] = v
                    var_order[id(v)] = slack_col = len(var_order)
                    _data.append(1)
                    _index.append(slack_col)
                rows.append(RowEntry(con, 1))
                con_data.append(np.array(_data))
                con_index.append(np.array(_index))
                con_index_ptr.append(con_index_ptr[-1] + len(_index))
            else:
                N = len(repn.linear)
                _data = np.fromiter(repn.linear.values(), float, N)
                _index = np.fromiter(map(var_order.__getitem__, repn.linear), float, N)
                if ub is not None:
                    rows.append(RowEntry(con, 1))
                    rhs.append(ub - offset)
                    con_data.append(_data)
                    con_index.append(_index)
                    con_index_ptr.append(con_index_ptr[-1] + N)
                if lb is not None:
                    rows.append(RowEntry(con, -1))
                    rhs.append(offset - lb)
                    con_data.append(-_data)
                    con_index.append(_index)
                    con_index_ptr.append(con_index_ptr[-1] + N)
        if with_debug_timing:
            timer.toc('Constraint %s', last_parent, level=logging.DEBUG)
        columns = list(var_map.values())
        c = scipy.sparse.csr_array((np.concatenate(obj_data), np.concatenate(obj_index), obj_index_ptr), [len(obj_index_ptr) - 1, len(columns)]).tocsc()
        A = scipy.sparse.csr_array((np.concatenate(con_data), np.concatenate(con_index), con_index_ptr), [len(rows), len(columns)]).tocsc()
        c_ip = c.indptr
        A_ip = A.indptr
        active_var_idx = list(filter(lambda i: A_ip[i] != A_ip[i + 1] or c_ip[i] != c_ip[i + 1], range(len(columns))))
        nCol = len(active_var_idx)
        if nCol != len(columns):
            columns = list(map(columns.__getitem__, active_var_idx))
            active_var_idx.append(c.indptr[-1])
            c = scipy.sparse.csc_array((c.data, c.indices, c.indptr.take(active_var_idx)), [c.shape[0], nCol])
            active_var_idx[-1] = A.indptr[-1]
            A = scipy.sparse.csc_array((A.data, A.indices, A.indptr.take(active_var_idx)), [A.shape[0], nCol])
        if self.config.nonnegative_vars:
            c, A, columns, eliminated_vars = _csc_to_nonnegative_vars(c, A, columns)
        else:
            eliminated_vars = []
        info = LinearStandardFormInfo(c, A, rhs, rows, columns, eliminated_vars)
        timer.toc('Generated linear standard form representation', delta=False)
        return info