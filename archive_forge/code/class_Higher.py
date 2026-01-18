from sympy.core.decorators import call_highest_priority
from sympy.core.expr import Expr
from sympy.core.mod import Mod
from sympy.core.numbers import Integer
from sympy.core.symbol import Symbol
from sympy.functions.elementary.integers import floor
class Higher(Integer):
    """
    Integer of value 1 and _op_priority 20

    Operations handled by this class return 1 and reverse operations return 2
    """
    _op_priority = 20.0
    result = 1

    def __new__(cls):
        obj = Expr.__new__(cls)
        obj.p = 1
        return obj

    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        return self.result

    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return 2 * self.result

    @call_highest_priority('__radd__')
    def __add__(self, other):
        return self.result

    @call_highest_priority('__add__')
    def __radd__(self, other):
        return 2 * self.result

    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        return self.result

    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        return 2 * self.result

    @call_highest_priority('__rpow__')
    def __pow__(self, other):
        return self.result

    @call_highest_priority('__pow__')
    def __rpow__(self, other):
        return 2 * self.result

    @call_highest_priority('__rtruediv__')
    def __truediv__(self, other):
        return self.result

    @call_highest_priority('__truediv__')
    def __rtruediv__(self, other):
        return 2 * self.result

    @call_highest_priority('__rmod__')
    def __mod__(self, other):
        return self.result

    @call_highest_priority('__mod__')
    def __rmod__(self, other):
        return 2 * self.result

    @call_highest_priority('__rfloordiv__')
    def __floordiv__(self, other):
        return self.result

    @call_highest_priority('__floordiv__')
    def __rfloordiv__(self, other):
        return 2 * self.result