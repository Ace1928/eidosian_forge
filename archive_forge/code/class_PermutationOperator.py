from collections import defaultdict
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.dense import zeros
from sympy.printing.str import StrPrinter
from sympy.utilities.iterables import has_dups
class PermutationOperator(Expr):
    """
    Represents the index permutation operator P(ij).

    P(ij)*f(i)*g(j) = f(i)*g(j) - f(j)*g(i)
    """
    is_commutative = True

    def __new__(cls, i, j):
        i, j = sorted(map(sympify, (i, j)), key=default_sort_key)
        obj = Basic.__new__(cls, i, j)
        return obj

    def get_permuted(self, expr):
        """
        Returns -expr with permuted indices.

        Explanation
        ===========

        >>> from sympy import symbols, Function
        >>> from sympy.physics.secondquant import PermutationOperator
        >>> p,q = symbols('p,q')
        >>> f = Function('f')
        >>> PermutationOperator(p,q).get_permuted(f(p,q))
        -f(q, p)

        """
        i = self.args[0]
        j = self.args[1]
        if expr.has(i) and expr.has(j):
            tmp = Dummy()
            expr = expr.subs(i, tmp)
            expr = expr.subs(j, i)
            expr = expr.subs(tmp, j)
            return S.NegativeOne * expr
        else:
            return expr

    def _latex(self, printer):
        return 'P(%s%s)' % self.args