from pyomo.core.expr.logical_expr import BooleanExpression
class PrecedenceExpression(BooleanExpression):

    def nargs(self):
        return 3

    @property
    def delay(self):
        return self._args_[2]

    def _to_string_impl(self, values, relation):
        delay = int(values[2])
        if delay == 0:
            first = values[0]
        elif delay > 0:
            first = '%s + %s' % (values[0], delay)
        else:
            first = '%s - %s' % (values[0], abs(delay))
        return '%s %s %s' % (first, relation, values[1])