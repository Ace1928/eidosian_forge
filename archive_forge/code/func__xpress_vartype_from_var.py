import logging
import os
import re
import sys
import time
from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.common.dependencies import attempt_import
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
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
def _xpress_vartype_from_var(self, var):
    """
        This function takes a pyomo variable and returns the appropriate xpress variable type
        :param var: pyomo.core.base.var.Var
        :return: xpress.continuous or xpress.binary or xpress.integer
        """
    if var.is_binary():
        vartype = xpress.binary
    elif var.is_integer():
        vartype = xpress.integer
    elif var.is_continuous():
        vartype = xpress.continuous
    else:
        raise ValueError('Variable domain type is not recognized for {0}'.format(var.domain))
    return vartype