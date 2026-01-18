from pyomo.gdp import GDP_Error
from pyomo.common.collections import ComponentSet
from pyomo.contrib.fbbt.expression_bounds_walker import ExpressionBoundsVisitor
import pyomo.contrib.fbbt.interval as interval
from pyomo.core import Suffix
def _get_M_from_args(self, constraint, bigMargs, arg_list, lower, upper):
    if bigMargs is None:
        return (lower, upper)
    need_lower = constraint.lower is not None
    need_upper = constraint.upper is not None
    parent = constraint.parent_component()
    if constraint in bigMargs:
        m = bigMargs[constraint]
        lower, upper, need_lower, need_upper = self._process_M_value(m, lower, upper, need_lower, need_upper, bigMargs, constraint, constraint, from_args=True)
        if not need_lower and (not need_upper):
            return (lower, upper)
    elif parent in bigMargs:
        m = bigMargs[parent]
        lower, upper, need_lower, need_upper = self._process_M_value(m, lower, upper, need_lower, need_upper, bigMargs, parent, constraint, from_args=True)
        if not need_lower and (not need_upper):
            return (lower, upper)
    for arg in arg_list:
        for block, val in arg.items():
            lower, upper, need_lower, need_upper = self._process_M_value(val, lower, upper, need_lower, need_upper, bigMargs, block, constraint, from_args=True)
            if not need_lower and (not need_upper):
                return (lower, upper)
    if None in bigMargs:
        m = bigMargs[None]
        lower, upper, need_lower, need_upper = self._process_M_value(m, lower, upper, need_lower, need_upper, bigMargs, None, constraint, from_args=True)
        if not need_lower and (not need_upper):
            return (lower, upper)
    return (lower, upper)