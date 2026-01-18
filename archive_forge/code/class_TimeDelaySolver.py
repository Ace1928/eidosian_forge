import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.core.base.set_types import NonNegativeIntegers
from pyomo.core.expr import (
from pyomo.contrib.pyros.util import (
from pyomo.contrib.pyros.util import replace_uncertain_bounds_with_constraints
from pyomo.contrib.pyros.util import get_vars_from_component
from pyomo.contrib.pyros.util import identify_objective_functions
from pyomo.common.collections import Bunch
import time
import math
from pyomo.contrib.pyros.util import time_code
from pyomo.contrib.pyros.uncertainty_sets import (
from pyomo.contrib.pyros.master_problem_methods import (
from pyomo.contrib.pyros.solve_data import MasterProblemData, ROSolveResults
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import scipy as sp, scipy_available
from pyomo.environ import maximize as pyo_max
from pyomo.common.errors import ApplicationError
from pyomo.opt import (
from pyomo.environ import (
import logging
from itertools import chain
class TimeDelaySolver(object):
    """
    Solver which puts program to sleep for a specified
    duration after having been invoked a specified number
    of times.
    """

    def __init__(self, calls_to_sleep, max_time, sub_solver):
        self.max_time = max_time
        self.calls_to_sleep = calls_to_sleep
        self.sub_solver = sub_solver
        self.num_calls = 0
        self.options = Bunch()

    def available(self, exception_flag=True):
        return True

    def license_is_valid(self):
        return True

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        pass

    def solve(self, model, **kwargs):
        """
        'Solve' a model.

        Parameters
        ----------
        model : ConcreteModel
            Model of interest.

        Returns
        -------
        results : SolverResults
            Solver results.
        """
        active_objs = [obj for obj in model.component_data_objects(Objective, active=True)]
        assert len(active_objs) == 1
        if self.num_calls < self.calls_to_sleep:
            results = self.sub_solver.solve(model, **kwargs)
            self.num_calls += 1
        else:
            time.sleep(self.max_time)
            results = SolverResults()
            self.num_calls = 0
            sol = Solution()
            sol.variable = {var.name: {'Value': value(var)} for var in model.component_data_objects(Var, active=True)}
            sol._cuid = False
            sol.status = SolutionStatus.stoppedByLimit
            results.solution.insert(sol)
            results.solver.time = self.max_time
            results.solver.termination_condition = TerminationCondition.maxTimeLimit
            results.solver.status = SolverStatus.warning
        return results