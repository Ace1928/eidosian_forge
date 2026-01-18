from pyomo.common.tempfiles import TempfileManager
from pyomo.common.fileutils import Executable
from pyomo.contrib.appsi.base import (
from pyomo.contrib.appsi.writers import NLWriter
from pyomo.common.log import LogStream
import logging
import subprocess
from pyomo.core.kernel.objective import minimize
import math
from pyomo.common.collections import ComponentMap
from pyomo.core.expr.numvalue import value
from pyomo.core.expr.visitor import replace_expressions
from typing import Optional, Sequence, NoReturn, List, Mapping
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.block import _BlockData
from pyomo.core.base.param import _ParamData
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.tee import TeeStream
import sys
from typing import Dict
from pyomo.common.config import ConfigValue, NonNegativeInt
from pyomo.common.errors import PyomoException
import os
from pyomo.contrib.appsi.cmodel import cmodel_available
from pyomo.core.staleflag import StaleFlagManager
def _parse_sol(self):
    solve_vars = self._writer.get_ordered_vars()
    solve_cons = self._writer.get_ordered_cons()
    results = Results()
    f = open(self._filename + '.sol', 'r')
    all_lines = list(f.readlines())
    f.close()
    termination_line = all_lines[1]
    if 'Optimal Solution Found' in termination_line:
        results.termination_condition = TerminationCondition.optimal
    elif 'Problem may be infeasible' in termination_line:
        results.termination_condition = TerminationCondition.infeasible
    elif 'problem might be unbounded' in termination_line:
        results.termination_condition = TerminationCondition.unbounded
    elif 'Maximum Number of Iterations Exceeded' in termination_line:
        results.termination_condition = TerminationCondition.maxIterations
    elif 'Maximum CPU Time Exceeded' in termination_line:
        results.termination_condition = TerminationCondition.maxTimeLimit
    else:
        results.termination_condition = TerminationCondition.unknown
    n_cons = len(solve_cons)
    n_vars = len(solve_vars)
    dual_lines = all_lines[12:12 + n_cons]
    primal_lines = all_lines[12 + n_cons:12 + n_cons + n_vars]
    rc_upper_info_line = all_lines[12 + n_cons + n_vars + 1]
    assert rc_upper_info_line.startswith('suffix')
    n_rc_upper = int(rc_upper_info_line.split()[2])
    assert 'ipopt_zU_out' in all_lines[12 + n_cons + n_vars + 2]
    upper_rc_lines = all_lines[12 + n_cons + n_vars + 3:12 + n_cons + n_vars + 3 + n_rc_upper]
    rc_lower_info_line = all_lines[12 + n_cons + n_vars + 3 + n_rc_upper]
    assert rc_lower_info_line.startswith('suffix')
    n_rc_lower = int(rc_lower_info_line.split()[2])
    assert 'ipopt_zL_out' in all_lines[12 + n_cons + n_vars + 3 + n_rc_upper + 1]
    lower_rc_lines = all_lines[12 + n_cons + n_vars + 3 + n_rc_upper + 2:12 + n_cons + n_vars + 3 + n_rc_upper + 2 + n_rc_lower]
    self._dual_sol = dict()
    self._primal_sol = ComponentMap()
    self._reduced_costs = ComponentMap()
    for ndx, dual in enumerate(dual_lines):
        dual = float(dual)
        con = solve_cons[ndx]
        self._dual_sol[con] = dual
    for ndx, primal in enumerate(primal_lines):
        primal = float(primal)
        var = solve_vars[ndx]
        self._primal_sol[var] = primal
    for rcu_line in upper_rc_lines:
        split_line = rcu_line.split()
        var_ndx = int(split_line[0])
        rcu = float(split_line[1])
        var = solve_vars[var_ndx]
        self._reduced_costs[var] = rcu
    for rcl_line in lower_rc_lines:
        split_line = rcl_line.split()
        var_ndx = int(split_line[0])
        rcl = float(split_line[1])
        var = solve_vars[var_ndx]
        if var in self._reduced_costs:
            if abs(rcl) > abs(self._reduced_costs[var]):
                self._reduced_costs[var] = rcl
        else:
            self._reduced_costs[var] = rcl
    for var in solve_vars:
        if var not in self._reduced_costs:
            self._reduced_costs[var] = 0
    if results.termination_condition == TerminationCondition.optimal and self.config.load_solution:
        for v, val in self._primal_sol.items():
            v.set_value(val, skip_validation=True)
        if self._writer.get_active_objective() is None:
            results.best_feasible_objective = None
        else:
            results.best_feasible_objective = value(self._writer.get_active_objective().expr)
    elif results.termination_condition == TerminationCondition.optimal:
        if self._writer.get_active_objective() is None:
            results.best_feasible_objective = None
        else:
            obj_expr_evaluated = replace_expressions(self._writer.get_active_objective().expr, substitution_map={id(v): val for v, val in self._primal_sol.items()}, descend_into_named_expressions=True, remove_named_expressions=True)
            results.best_feasible_objective = value(obj_expr_evaluated)
    elif self.config.load_solution:
        raise RuntimeError('A feasible solution was not found, so no solution can be loaded. If using the appsi.solvers.Ipopt interface, you can set opt.config.load_solution=False. If using the environ.SolverFactory interface, you can set opt.solve(model, load_solutions = False). Then you can check results.termination_condition and results.best_feasible_objective before loading a solution.')
    return results