from pyomo.common.tempfiles import TempfileManager
from pyomo.common.fileutils import Executable
from pyomo.contrib.appsi.base import (
from pyomo.contrib.appsi.writers import LPWriter
from pyomo.common.log import LogStream
import logging
import subprocess
from pyomo.core.kernel.objective import minimize, maximize
import math
from pyomo.common.collections import ComponentMap
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
from pyomo.contrib.appsi.cmodel import cmodel_available
from pyomo.core.staleflag import StaleFlagManager
def _parse_soln(self):
    results = Results()
    f = open(self._filename + '.soln', 'r')
    all_lines = list(f.readlines())
    f.close()
    termination_line = all_lines[0].lower()
    obj_val = None
    if termination_line.startswith('optimal'):
        results.termination_condition = TerminationCondition.optimal
        obj_val = float(termination_line.split()[-1])
    elif 'infeasible' in termination_line:
        results.termination_condition = TerminationCondition.infeasible
    elif 'unbounded' in termination_line:
        results.termination_condition = TerminationCondition.unbounded
    elif termination_line.startswith('stopped on time'):
        results.termination_condition = TerminationCondition.maxTimeLimit
        obj_val = float(termination_line.split()[-1])
    elif termination_line.startswith('stopped on iterations'):
        results.termination_condition = TerminationCondition.maxIterations
        obj_val = float(termination_line.split()[-1])
    else:
        results.termination_condition = TerminationCondition.unknown
    first_con_line = None
    last_con_line = None
    first_var_line = None
    last_var_line = None
    for ndx, line in enumerate(all_lines):
        if line.strip('*').strip().startswith('0'):
            if first_con_line is None:
                first_con_line = ndx
            else:
                last_con_line = ndx - 1
                first_var_line = ndx
    last_var_line = len(all_lines) - 1
    self._dual_sol = dict()
    self._primal_sol = dict()
    self._reduced_costs = dict()
    symbol_map = self._writer.symbol_map
    for line in all_lines[first_con_line:last_con_line + 1]:
        split_line = line.strip('*')
        split_line = split_line.split()
        name = split_line[1]
        orig_name = name[:-3]
        if orig_name == 'obj_const_con':
            continue
        con = symbol_map.bySymbol[orig_name]
        dual_val = float(split_line[-1])
        if con in self._dual_sol:
            if abs(dual_val) > abs(self._dual_sol[con]):
                self._dual_sol[con] = dual_val
        else:
            self._dual_sol[con] = dual_val
    for line in all_lines[first_var_line:last_var_line + 1]:
        split_line = line.strip('*')
        split_line = split_line.split()
        name = split_line[1]
        if name == 'obj_const':
            continue
        val = float(split_line[2])
        rc = float(split_line[3])
        var = symbol_map.bySymbol[name]
        self._primal_sol[id(var)] = (var, val)
        self._reduced_costs[id(var)] = (var, rc)
    if self.version() < (2, 10, 2) and self._writer.get_active_objective() is not None and (self._writer.get_active_objective().sense == maximize):
        if obj_val is not None:
            obj_val = -obj_val
        for con, dual_val in self._dual_sol.items():
            self._dual_sol[con] = -dual_val
        for v_id, (v, rc_val) in self._reduced_costs.items():
            self._reduced_costs[v_id] = (v, -rc_val)
    if results.termination_condition == TerminationCondition.optimal and self.config.load_solution:
        for v_id, (v, val) in self._primal_sol.items():
            v.set_value(val, skip_validation=True)
        if self._writer.get_active_objective() is None:
            results.best_feasible_objective = None
        else:
            results.best_feasible_objective = obj_val
    elif results.termination_condition == TerminationCondition.optimal:
        if self._writer.get_active_objective() is None:
            results.best_feasible_objective = None
        else:
            results.best_feasible_objective = obj_val
    elif self.config.load_solution:
        raise RuntimeError('A feasible solution was not found, so no solution can be loaded.Please set opt.config.load_solution=False and check results.termination_condition and results.best_feasible_objective before loading a solution.')
    return results