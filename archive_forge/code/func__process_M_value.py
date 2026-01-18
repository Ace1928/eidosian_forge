from pyomo.gdp import GDP_Error
from pyomo.common.collections import ComponentSet
from pyomo.contrib.fbbt.expression_bounds_walker import ExpressionBoundsVisitor
import pyomo.contrib.fbbt.interval as interval
from pyomo.core import Suffix
def _process_M_value(self, m, lower, upper, need_lower, need_upper, src, key, constraint, from_args=False):
    m = _convert_M_to_tuple(m, constraint)
    if need_lower and m[0] is not None:
        if from_args:
            self.used_args[key] = m
        lower = (m[0], src, key)
        need_lower = False
    if need_upper and m[1] is not None:
        if from_args:
            self.used_args[key] = m
        upper = (m[1], src, key)
        need_upper = False
    return (lower, upper, need_lower, need_upper)