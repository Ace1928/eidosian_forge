from pyomo.contrib.cp.interval_var import (
from pyomo.core.base.component import Component
from pyomo.core.expr.base import ExpressionBase
from pyomo.core.expr.logical_expr import BooleanExpression
@property
def _interval_var(self):
    return self._args_[0]