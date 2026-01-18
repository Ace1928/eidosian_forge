from collections.abc import Iterable
import logging
from pyomo.common.collections import ComponentSet
from pyomo.common.config import (
from pyomo.common.errors import ApplicationError, PyomoException
from pyomo.core.base import Var, _VarData
from pyomo.core.base.param import Param, _ParamData
from pyomo.opt import SolverFactory
from pyomo.contrib.pyros.util import ObjectiveType, setup_pyros_logger
from pyomo.contrib.pyros.uncertainty_sets import UncertaintySet
class SolverNotResolvable(PyomoException):
    """
    Exception type for failure to cast an object to a Pyomo solver.
    """