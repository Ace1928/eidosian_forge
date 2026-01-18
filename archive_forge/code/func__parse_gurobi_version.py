import logging
import re
import sys
from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.common.dependencies import attempt_import
from pyomo.common.errors import ApplicationError
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.tee import capture_output
from pyomo.core.expr.numvalue import is_fixed
from pyomo.core.expr.numvalue import value
from pyomo.core.staleflag import StaleFlagManager
from pyomo.repn import generate_standard_repn
from pyomo.solvers.plugins.solvers.direct_solver import DirectSolver
from pyomo.solvers.plugins.solvers.direct_or_persistent_solver import (
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.opt.results.results_ import SolverResults
from pyomo.opt.results.solution import Solution, SolutionStatus
from pyomo.opt.results.solver import TerminationCondition, SolverStatus
from pyomo.opt.base import SolverFactory
from pyomo.core.base.suffix import Suffix
import pyomo.core.base.var
def _parse_gurobi_version(gurobipy, avail):
    if not avail:
        return
    GurobiDirect._version = gurobipy.gurobi.version()
    GurobiDirect._name = 'Gurobi %s.%s%s' % GurobiDirect._version
    while len(GurobiDirect._version) < 4:
        GurobiDirect._version += (0,)
    GurobiDirect._version = GurobiDirect._version[:4]
    GurobiDirect._version_major = GurobiDirect._version[0]