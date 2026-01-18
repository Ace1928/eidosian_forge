import collections
import numbers
class ProductCst(LinearExpr):
    """Represents the product of a LinearExpr by a constant."""

    def __init__(self, expr, coef):
        self.__expr = CastToLinExp(expr)
        if isinstance(coef, numbers.Number):
            self.__coef = coef
        else:
            raise TypeError

    def __str__(self):
        if self.__coef == -1:
            return '-' + str(self.__expr)
        else:
            return '(' + str(self.__coef) + ' * ' + str(self.__expr) + ')'

    def AddSelfToCoeffMapOrStack(self, coeffs, multiplier, stack):
        current_multiplier = multiplier * self.__coef
        if current_multiplier:
            stack.append((current_multiplier, self.__expr))