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
def _extract_jac(self, m):
    """
        Extract jacobian from the stochastic program

        Parameters
        ----------
        m: solved stochastic program model

        Returns
        -------
        JAC: the overall jacobian as a dictionary
        """
    jac = {}
    for p in self.param.keys():
        jac_para = []
        for res in m.measured_variables:
            jac_para.append(pyo.value(m.sensitivity_jacobian[p, res]))
        jac[p] = jac_para
    return jac