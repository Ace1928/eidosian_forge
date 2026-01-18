import logging
import re
import sys
import csv
import subprocess
from pyomo.common.tempfiles import TempfileManager
from pyomo.common import Executable
from pyomo.common.collections import Bunch
from pyomo.opt import (
from pyomo.opt.base.solvers import _extract_version
from pyomo.opt.solver import SystemCallSolver
from pyomo.solvers.mockmip import MockMIP
def _process_soln_mip(self, row, reader, results, obj_name, variable_names, constraint_names):
    """
        Process a basic solution
        """
    status = row[4]
    obj_val = float(row[5])
    solv = results.solver
    if status == 'n':
        solv.termination_condition = TerminationCondition.infeasible
    elif status == 'u':
        if solv.termination_condition == TerminationCondition.unknown:
            solv.termination_condition = TerminationCondition.other
    elif status == 'f' or status == 'o':
        if obj_val > 1e+18 or obj_val < -1e+18:
            solv.termination_condition = TerminationCondition.unbounded
            return
        soln = results.solution.add()
        if status == 'f':
            soln.status = SolutionStatus.feasible
            solv.termination_condition = TerminationCondition.feasible
        else:
            soln.status = SolutionStatus.optimal
            solv.termination_condition = TerminationCondition.optimal
        if status == 'o':
            soln.gap = 0.0
            results.problem.lower_bound = obj_val
            results.problem.upper_bound = obj_val
        soln.objective[obj_name] = {'Value': obj_val}
        while True:
            row = next(reader)
            if len(row) == 0:
                break
            rtype = row[0]
            if rtype == 'i':
                continue
            elif rtype == 'j':
                rtype, cid, cval = row
                vname = variable_names[int(cid)]
                if 'ONE_VAR_CONSTANT' == vname:
                    continue
                soln.variable[vname] = {'Value': float(cval)}
            elif rtype == 'e':
                break
            elif rtype == 'c':
                continue
            else:
                raise ValueError('Unexpected row type: ' + rtype)