from pyomo.contrib.cp.interval_var import (
from pyomo.core.base.component import Component
from pyomo.core.expr.base import ExpressionBase
from pyomo.core.expr.logical_expr import BooleanExpression
class AlwaysIn(BooleanExpression):
    """
    An expression representing the constraint that a cumulative function is
    required to take values within a tuple of bounds over a specified time
    interval. (Often used to enforce limits on resource availability.)

    Args:
        cumul_func (CumulativeFunction): Step function being constrained
        bounds (tuple of two integers): Lower and upper bounds to enforce on
            the cumulative function
        times (tuple of two integers): The time interval (start, end) over
            which to enforce the bounds on the values of the cumulative
            function.
    """
    __slots__ = ()

    def __init__(self, args=None, cumul_func=None, bounds=None, times=None):
        if args:
            if any((arg is not None for arg in {cumul_func, bounds, times})):
                raise ValueError('Cannot specify both args and any of {cumul_func, bounds, times}')
            self._args_ = args
        else:
            self._args_ = (cumul_func, bounds[0], bounds[1], times[0], times[1])

    def nargs(self):
        return 5

    def _to_string(self, values, verbose, smap):
        return '(%s).within(bounds=(%s, %s), times=(%s, %s))' % (values[0], values[1], values[2], values[3], values[4])