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
def _sgn(self, p):
    """
        This is a helper function for stochastic_program function to compute the determinant formula.
        Give the signature of a permutation

        Parameters
        -----------
        p: the permutation (a list)

        Returns
        -------
        1 if the number of exchange is an even number
        -1 if the number is an odd number
        """
    if len(p) == 1:
        return 1
    trans = 0
    for i in range(0, len(p)):
        j = i + 1
        for j in range(j, len(p)):
            if p[i] > p[j]:
                trans = trans + 1
    if trans % 2 == 0:
        return 1
    else:
        return -1