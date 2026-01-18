import logging
import unittest
from pyomo.core.base import ConcreteModel, Var, _VarData
from pyomo.common.log import LoggingIntercept
from pyomo.common.errors import ApplicationError
from pyomo.core.base.param import Param, _ParamData
from pyomo.contrib.pyros.config import (
from pyomo.contrib.pyros.util import ObjectiveType
from pyomo.opt import SolverFactory, SolverResults
from pyomo.contrib.pyros.uncertainty_sets import BoxSet
from pyomo.common.dependencies import numpy_available
class AvailableSolver:
    """
    Perennially available placeholder solver.
    """

    def available(self, exception_flag=False):
        """
        Check solver available.
        """
        return True

    def solve(self, model, **kwds):
        """
        Return SolverResults object with 'unknown' termination
        condition. Model remains unchanged.
        """
        return SolverResults()