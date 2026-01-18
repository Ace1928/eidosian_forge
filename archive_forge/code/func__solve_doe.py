from pyomo.common.dependencies import numpy as np, numpy_available
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pickle
from itertools import permutations, product
import logging
from enum import Enum
from pyomo.common.timing import TicTocTimer
from pyomo.contrib.sensitivity_toolbox.sens import get_dsdp
from pyomo.contrib.doe.scenario import ScenarioGenerator, FiniteDifferenceStep
from pyomo.contrib.doe.result import FisherResults, GridSearchResult
def _solve_doe(self, m, fix=False, opt_option=None):
    """Solve DOE model.
        If it's a square problem, fix design variable and solve.
        Else, fix design variable and solve square problem firstly, then unfix them and solve the optimization problem

        Parameters
        ----------
        m:model
        fix: if true, solve two times (square first). Else, just solve the square problem
        opt_option: a dictionary, keys are design variable name, values are True or False,
            deciding if this design variable is optimized as DOF this time.
            If None, all design variables are optimized as DOF this time.

        Returns
        -------
        solver_results: solver results
        """
    mod = self._fix_design(m, self.design_values, fix_opt=fix, optimize_option=opt_option)
    solver_result = self.solver.solve(mod, tee=self.tee_opt)
    return solver_result