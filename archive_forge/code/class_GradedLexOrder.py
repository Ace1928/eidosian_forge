from __future__ import annotations
from sympy.core import Symbol
from sympy.utilities.iterables import iterable
class GradedLexOrder(MonomialOrder):
    """Graded lexicographic order of monomials. """
    alias = 'grlex'
    is_global = True

    def __call__(self, monomial):
        return (sum(monomial), monomial)