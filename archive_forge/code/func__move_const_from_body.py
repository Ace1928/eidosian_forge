import pyomo.common.unittest as unittest
from pyomo.common.errors import MouseTrap, DeveloperError
from pyomo.common.log import LoggingIntercept
import logging
from pyomo.core.expr.sympy_tools import sympy_available
from pyomo.core.plugins.transform.logical_to_linear import (
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.repn import generate_standard_repn
from io import StringIO
def _move_const_from_body(lower, repn, upper):
    if repn.constant is not None and (not repn.constant == 0):
        if lower is not None:
            lower -= repn.constant
        if upper is not None:
            upper -= repn.constant
    return (value(lower), repn, value(upper))