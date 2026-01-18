from sympy.core import Basic, Integer
import operator
def _compare_term(self, other, op):
    if self.exp == other.exp:
        return op(self.mult, other.mult)
    else:
        return op(self.exp, other.exp)