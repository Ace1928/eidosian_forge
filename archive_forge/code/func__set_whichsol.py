import logging
import re
import sys
import itertools
import operator
import pyomo.core.base.var
import pyomo.core.base.constraint
from pyomo.common.dependencies import attempt_import
from pyomo.common.tempfiles import TempfileManager
from pyomo.core import is_fixed, value, minimize, maximize
from pyomo.core.base.suffix import Suffix
from pyomo.core.staleflag import StaleFlagManager
from pyomo.opt.base.solvers import OptSolver
from pyomo.repn import generate_standard_repn
from pyomo.solvers.plugins.solvers.direct_solver import DirectSolver
from pyomo.solvers.plugins.solvers.direct_or_persistent_solver import (
from pyomo.common.collections import ComponentMap, ComponentSet, Bunch
from pyomo.opt import SolverFactory
from pyomo.core.kernel.conic import (
from pyomo.opt.results.results_ import SolverResults
from pyomo.opt.results.solution import Solution, SolutionStatus
from pyomo.opt.results.solver import TerminationCondition, SolverStatus
def _set_whichsol(self):
    itr_soltypes = [mosek.problemtype.qo, mosek.problemtype.qcqo, mosek.problemtype.conic]
    if self._solver_model.getnumintvar() >= 1:
        self._whichsol = mosek.soltype.itg
    elif self._solver_model.getprobtype() in itr_soltypes:
        self._whichsol = mosek.soltype.itr
    elif self._solver_model.getprobtype() == mosek.problemtype.lo:
        self._whichsol = mosek.soltype.bas