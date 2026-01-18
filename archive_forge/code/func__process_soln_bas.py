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
def _process_soln_bas(self, row, reader, results, obj_name, variable_names, constraint_names):
    """
        Process a basic solution
        """
    prows = int(row[2])
    pcols = int(row[3])
    pstat = row[4]
    dstat = row[5]
    obj_val = float(row[6])
    solv = results.solver
    if dstat == 'n':
        solv.termination_condition = TerminationCondition.unbounded
    elif pstat == 'n':
        solv.termination_condition = TerminationCondition.unbounded
    elif pstat == 'i':
        solv.termination_condition = TerminationCondition.infeasible
    elif pstat == 'u':
        if solv.termination_condition == TerminationCondition.unknown:
            solv.termination_condition = TerminationCondition.other
    elif pstat == 'f':
        soln = results.solution.add()
        soln.status = SolutionStatus.feasible
        solv.termination_condition = TerminationCondition.optimal
        soln.gap = 0.0
        results.problem.lower_bound = obj_val
        results.problem.upper_bound = obj_val
        soln.objective[obj_name] = {'Value': obj_val}
        extract_duals = False
        extract_reduced_costs = False
        for suffix in self._suffixes:
            if re.match(suffix, 'dual'):
                extract_duals = True
            elif re.match(suffix, 'rc'):
                extract_reduced_costs = True
        range_duals = {}
        while True:
            row = next(reader)
            if len(row) == 0:
                break
            rtype = row[0]
            if rtype == 'i':
                if not extract_duals:
                    continue
                rtype, rid, rst, rprim, rdual = row
                cname = constraint_names[int(rid)]
                if 'ONE_VAR_CONSTANT' == cname[-16:]:
                    continue
                rdual = float(rdual)
                if cname.startswith('c_'):
                    soln.constraint[cname] = {'Dual': rdual}
                elif cname.startswith('r_l_'):
                    range_duals.setdefault(cname[4:], [0, 0])[0] = rdual
                elif cname.startswith('r_u_'):
                    range_duals.setdefault(cname[4:], [0, 0])[1] = rdual
            elif rtype == 'j':
                rtype, cid, cst, cprim, cdual = row
                vname = variable_names[int(cid)]
                if 'ONE_VAR_CONSTANT' == vname:
                    continue
                cprim = float(cprim)
                if extract_reduced_costs is False:
                    soln.variable[vname] = {'Value': cprim}
                else:
                    soln.variable[vname] = {'Value': cprim, 'Rc': float(cdual)}
            elif rtype == 'e':
                break
            elif rtype == 'c':
                continue
            else:
                raise ValueError('Unexpected row type: ' + rtype)
        scon = soln.Constraint
        for key, (ld, ud) in range_duals.items():
            if abs(ld) > abs(ud):
                scon['r_l_' + key] = {'Dual': ld}
            else:
                scon['r_l_' + key] = {'Dual': ud}