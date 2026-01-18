import operator
from pyomo.common.deprecation import deprecated
from pyomo.common.errors import PyomoException, DeveloperError
from pyomo.common.numeric_types import (
from .base import ExpressionBase
from .boolean_value import BooleanValue
from .expr_common import _lt, _le, _eq, ExpressionType
from .numvalue import is_potentially_variable, is_constant
from .visitor import polynomial_degree
class RelationalExpression(ExpressionBase, BooleanValue):
    __slots__ = ('_args_',)
    EXPRESSION_SYSTEM = ExpressionType.RELATIONAL

    def __init__(self, args):
        self._args_ = args

    def __bool__(self):
        if self.is_constant():
            return bool(self())
        raise PyomoException('\nCannot convert non-constant Pyomo expression (%s) to bool.\nThis error is usually caused by using a Var, unit, or mutable Param in a\nBoolean context such as an "if" statement, or when checking container\nmembership or equality. For example,\n    >>> m.x = Var()\n    >>> if m.x >= 1:\n    ...     pass\nand\n    >>> m.y = Var()\n    >>> if m.y in [m.x, m.y]:\n    ...     pass\nwould both cause this exception.'.strip() % (self,))

    @property
    def args(self):
        """
        Return the child nodes

        Returns: Either a list or tuple (depending on the node storage
            model) containing only the child nodes of this node
        """
        return self._args_[:self.nargs()]

    @deprecated('is_relational() is deprecated in favor of is_expression_type(ExpressionType.RELATIONAL)', version='6.4.3')
    def is_relational(self):
        return self.is_expression_type(ExpressionType.RELATIONAL)

    def is_potentially_variable(self):
        return any((is_potentially_variable(arg) for arg in self._args_))

    def polynomial_degree(self):
        """
        Return the polynomial degree of the expression.

        Returns:
            A non-negative integer that is the polynomial
            degree if the expression is polynomial, or :const:`None` otherwise.
        """
        return polynomial_degree(self)

    def _compute_polynomial_degree(self, result):
        ans = 0
        for x in result:
            if x is None:
                return None
            elif ans < x:
                ans = x
        return ans

    def __eq__(self, other):
        """
        Equal to operator

        This method is called when Python processes statements of the form::

            self == other
            other == self
        """
        return _generate_relational_expression(_eq, self, other)

    def __lt__(self, other):
        """
        Less than operator

        This method is called when Python processes statements of the form::

            self < other
            other > self
        """
        return _generate_relational_expression(_lt, self, other)

    def __gt__(self, other):
        """
        Greater than operator

        This method is called when Python processes statements of the form::

            self > other
            other < self
        """
        return _generate_relational_expression(_lt, other, self)

    def __le__(self, other):
        """
        Less than or equal operator

        This method is called when Python processes statements of the form::

            self <= other
            other >= self
        """
        return _generate_relational_expression(_le, self, other)

    def __ge__(self, other):
        """
        Greater than or equal operator

        This method is called when Python processes statements of the form::

            self >= other
            other <= self
        """
        return _generate_relational_expression(_le, other, self)