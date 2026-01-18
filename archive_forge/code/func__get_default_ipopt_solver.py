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
def _get_default_ipopt_solver(self):
    """Default solver"""
    solver = SolverFactory('ipopt')
    solver.options['linear_solver'] = 'ma57'
    solver.options['halt_on_ampl_error'] = 'yes'
    solver.options['max_iter'] = 3000
    return solver