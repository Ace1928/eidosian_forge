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
def _compute_stochastic_program(self, m, optimize_option):
    """
        Solve the stochastic program problem as a square problem.
        """
    result_square = self._solve_doe(m, fix=True, opt_option=optimize_option)
    jac_square = self._extract_jac(m)
    analysis_square = FisherResults(list(self.param.keys()), self.measurement_vars, jacobian_info=None, all_jacobian_info=jac_square, prior_FIM=self.prior_FIM, scale_constant_value=self.scale_constant_value)
    analysis_square.result_analysis(result=result_square)
    analysis_square.model = m
    self.analysis_square = analysis_square
    return (m, analysis_square)