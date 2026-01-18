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
def _glpk_get_solution_status(self, status):
    if GLP_FEAS == status:
        return SolutionStatus.feasible
    elif GLP_INFEAS == status:
        return SolutionStatus.infeasible
    elif GLP_NOFEAS == status:
        return SolutionStatus.infeasible
    elif GLP_UNDEF == status:
        return SolutionStatus.other
    elif GLP_OPT == status:
        return SolutionStatus.optimal
    raise RuntimeError('Unknown solution status returned by GLPK solver')