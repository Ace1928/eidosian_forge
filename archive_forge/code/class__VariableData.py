import logging
import re
import sys
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.core.base import Suffix, Var, Constraint, SOSConstraint, Objective
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
import time
class _VariableData(object):

    def __init__(self, solver_model):
        self._solver_model = solver_model
        self.lb = []
        self.ub = []
        self.types = []
        self.names = []

    def add(self, lb, ub, type_, name):
        self.lb.append(lb)
        self.ub.append(ub)
        self.types.append(type_)
        self.names.append(name)

    def store_in_cplex(self):
        self._solver_model.variables.add(lb=self.lb, ub=self.ub, types=self.types, names=self.names)