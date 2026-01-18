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
def _generate_boolean_model(nvars):
    m = ConcreteModel()
    m.s = RangeSet(nvars)
    m.Y = BooleanVar(m.s)
    m.constraint = LogicalConstraint(expr=exactly(2, m.Y))
    return m