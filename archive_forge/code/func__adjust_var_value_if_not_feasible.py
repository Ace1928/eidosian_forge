from math import fabs
import math
from pyomo.core.base.transformation import TransformationFactory
from pyomo.common.config import (
from pyomo.core.base.constraint import Constraint
from pyomo.core.expr.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.repn import generate_standard_repn
def _adjust_var_value_if_not_feasible(var):
    var_value = var.value
    if var.has_lb():
        var_value = max(var_value, var.lb)
    if var.has_ub():
        var_value = min(var_value, var.ub)
    if var.is_integer():
        var.set_value(int(var_value))
    else:
        var.set_value(var_value)