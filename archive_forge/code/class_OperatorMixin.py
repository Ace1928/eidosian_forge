from ..utils import SchemaBase
class OperatorMixin:

    def _to_expr(self):
        return repr(self)

    def _from_expr(self, expr):
        return expr

    def __add__(self, other):
        comp_value = BinaryExpression('+', self, other)
        return self._from_expr(comp_value)

    def __radd__(self, other):
        comp_value = BinaryExpression('+', other, self)
        return self._from_expr(comp_value)

    def __sub__(self, other):
        comp_value = BinaryExpression('-', self, other)
        return self._from_expr(comp_value)

    def __rsub__(self, other):
        comp_value = BinaryExpression('-', other, self)
        return self._from_expr(comp_value)

    def __mul__(self, other):
        comp_value = BinaryExpression('*', self, other)
        return self._from_expr(comp_value)

    def __rmul__(self, other):
        comp_value = BinaryExpression('*', other, self)
        return self._from_expr(comp_value)

    def __truediv__(self, other):
        comp_value = BinaryExpression('/', self, other)
        return self._from_expr(comp_value)

    def __rtruediv__(self, other):
        comp_value = BinaryExpression('/', other, self)
        return self._from_expr(comp_value)
    __div__ = __truediv__
    __rdiv__ = __rtruediv__

    def __mod__(self, other):
        comp_value = BinaryExpression('%', self, other)
        return self._from_expr(comp_value)

    def __rmod__(self, other):
        comp_value = BinaryExpression('%', other, self)
        return self._from_expr(comp_value)

    def __pow__(self, other):
        comp_value = FunctionExpression('pow', (self, other))
        return self._from_expr(comp_value)

    def __rpow__(self, other):
        comp_value = FunctionExpression('pow', (other, self))
        return self._from_expr(comp_value)

    def __neg__(self):
        comp_value = UnaryExpression('-', self)
        return self._from_expr(comp_value)

    def __pos__(self):
        comp_value = UnaryExpression('+', self)
        return self._from_expr(comp_value)

    def __eq__(self, other):
        comp_value = BinaryExpression('===', self, other)
        return self._from_expr(comp_value)

    def __ne__(self, other):
        comp_value = BinaryExpression('!==', self, other)
        return self._from_expr(comp_value)

    def __gt__(self, other):
        comp_value = BinaryExpression('>', self, other)
        return self._from_expr(comp_value)

    def __lt__(self, other):
        comp_value = BinaryExpression('<', self, other)
        return self._from_expr(comp_value)

    def __ge__(self, other):
        comp_value = BinaryExpression('>=', self, other)
        return self._from_expr(comp_value)

    def __le__(self, other):
        comp_value = BinaryExpression('<=', self, other)
        return self._from_expr(comp_value)

    def __abs__(self):
        comp_value = FunctionExpression('abs', (self,))
        return self._from_expr(comp_value)

    def __and__(self, other):
        comp_value = BinaryExpression('&&', self, other)
        return self._from_expr(comp_value)

    def __rand__(self, other):
        comp_value = BinaryExpression('&&', other, self)
        return self._from_expr(comp_value)

    def __or__(self, other):
        comp_value = BinaryExpression('||', self, other)
        return self._from_expr(comp_value)

    def __ror__(self, other):
        comp_value = BinaryExpression('||', other, self)
        return self._from_expr(comp_value)

    def __invert__(self):
        comp_value = UnaryExpression('!', self)
        return self._from_expr(comp_value)