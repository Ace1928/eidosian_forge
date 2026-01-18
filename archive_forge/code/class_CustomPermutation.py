from itertools import permutations
from sympy.core.expr import unchanged
from sympy.core.numbers import Integer
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol
from sympy.core.singleton import S
from sympy.combinatorics.permutations import \
from sympy.printing import sstr, srepr, pretty, latex
from sympy.testing.pytest import raises, warns_deprecated_sympy
class CustomPermutation(Permutation):

    def __call__(self, *i):
        try:
            return super().__call__(*i)
        except TypeError:
            pass
        try:
            perm_obj = i[0]
            return [self._array_form[j] for j in perm_obj]
        except TypeError:
            raise TypeError('unrecognized argument')

    def __eq__(self, other):
        if isinstance(other, Permutation):
            return self._hashable_content() == other._hashable_content()
        else:
            return super().__eq__(other)

    def __hash__(self):
        return super().__hash__()