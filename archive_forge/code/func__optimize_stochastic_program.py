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
def _optimize_stochastic_program(self, m):
    """
        Solve the stochastic program problem as an optimization problem.
        """
    m = self._add_objective(m)
    result_doe = self._solve_doe(m, fix=False)
    jac_optimize = self._extract_jac(m)
    analysis_optimize = FisherResults(list(self.param.keys()), self.measurement_vars, jacobian_info=None, all_jacobian_info=jac_optimize, prior_FIM=self.prior_FIM)
    analysis_optimize.result_analysis(result=result_doe)
    analysis_optimize.model = m
    return analysis_optimize