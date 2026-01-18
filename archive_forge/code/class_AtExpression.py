from pyomo.core.expr.logical_expr import BooleanExpression
class AtExpression(PrecedenceExpression):
    """
    Base class for all precedence expressions.

    args:
        args (tuple): child nodes of type IntervalVar. We expect them to be
                      (first_time, second_time, delay).
        delay: A (possibly negative) integer value representing the number of
               time periods delay in the precedence relationship
    """

    def _to_string(self, values, verbose, smap):
        return self._to_string_impl(values, '==')