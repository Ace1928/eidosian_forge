from pyomo.contrib.cp.interval_var import (
from pyomo.core.base.component import Component
from pyomo.core.expr.base import ExpressionBase
from pyomo.core.expr.logical_expr import BooleanExpression
def _sum_unit_and_cumul(_unit, _cumul):
    return CumulativeFunction([_unit] + _cumul.args)