from pyomo.contrib.cp.interval_var import (
from pyomo.core.base.component import Component
from pyomo.core.expr.base import ExpressionBase
from pyomo.core.expr.logical_expr import BooleanExpression
def _generate_sum_expression(_self, _other):
    if isinstance(_self, CumulativeFunction):
        if isinstance(_other, CumulativeFunction):
            return _sum_cumuls(_self, _other)
        elif isinstance(_other, StepFunction):
            return _sum_cumul_and_unit(_self, _other)
    elif isinstance(_self, StepFunction):
        if isinstance(_other, CumulativeFunction):
            return _sum_unit_and_cumul(_self, _other)
        elif isinstance(_other, StepFunction):
            return _sum_two_units(_self, _other)
    raise TypeError('Cannot add object of class %s to object of class %s' % (_other.__class__, _self.__class__))