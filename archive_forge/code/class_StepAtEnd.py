from pyomo.contrib.cp.interval_var import (
from pyomo.core.base.component import Component
from pyomo.core.expr.base import ExpressionBase
from pyomo.core.expr.logical_expr import BooleanExpression
class StepAtEnd(StepBase):
    __slots__ = ()

    def _to_string(self, values, verbose, smap):
        return 'Step(%s, height=%s)' % (self._time.end_time, values[1])