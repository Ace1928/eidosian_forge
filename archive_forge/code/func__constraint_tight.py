from pyomo.common.config import (
from pyomo.common.modeling import unique_component_name
from pyomo.core import (
from pyomo.core.expr import differentiate
from pyomo.common.collections import ComponentSet
from pyomo.opt import SolverFactory
from pyomo.repn import generate_standard_repn
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import (
from pyomo.contrib.fme.fourier_motzkin_elimination import (
import logging
def _constraint_tight(model, constraint, TOL):
    """
    Returns a list [a,b] where a is -1 if the lower bound is tight or
    slightly violated, b is 1 if the upper bound is tight of slightly
    violated, and [a,b]=[-1,1] if we have an exactly satisfied (or
    slightly violated) equality.
    """
    val = value(constraint.body)
    ans = [0, 0]
    if constraint.lower is not None:
        if val - value(constraint.lower) <= TOL:
            ans[0] -= 1
    if constraint.upper is not None:
        if value(constraint.upper) - val <= TOL:
            ans[1] += 1
    return ans