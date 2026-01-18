from contextlib import contextmanager
import logging
from math import fabs
import sys
from pyomo.common import timing
from pyomo.common.collections import ComponentSet
from pyomo.common.deprecation import deprecation_warning
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.contrib.mcpp.pyomo_mcpp import mcpp_available, McCormick
from pyomo.core import (
from pyomo.core.expr.numvalue import native_types
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import _parent_disjunct
from pyomo.opt import SolverFactory
class SuppressInfeasibleWarning(object):
    """Suppress the infeasible model warning message from solve().

    The "WARNING: Loading a SolverResults object with a warning status" warning
    message from calling solve() is often unwanted, but there is no clear way
    to suppress it.

    This is modeled on LoggingIntercept from pyomo.common.log,
    but different in function.

    """

    class InfeasibleWarningFilter(logging.Filter):

        def filter(self, record):
            return not record.getMessage().startswith('Loading a SolverResults object with a warning status into model')
    warning_filter = InfeasibleWarningFilter()

    def __enter__(self):
        logger = logging.getLogger('pyomo.core')
        logger.addFilter(self.warning_filter)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        logger = logging.getLogger('pyomo.core')
        logger.removeFilter(self.warning_filter)