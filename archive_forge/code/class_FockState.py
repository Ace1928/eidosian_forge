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
class FockState(Expr):
    """
    Many particle Fock state with a sequence of occupation numbers.

    Anywhere you can have a FockState, you can also have S.Zero.
    All code must check for this!

    Base class to represent FockStates.
    """
    is_commutative = False

    def __new__(cls, occupations):
        """
        occupations is a list with two possible meanings:

        - For bosons it is a list of occupation numbers.
          Element i is the number of particles in state i.

        - For fermions it is a list of occupied orbits.
          Element 0 is the state that was occupied first, element i
          is the i'th occupied state.
        """
        occupations = list(map(sympify, occupations))
        obj = Basic.__new__(cls, Tuple(*occupations))
        return obj

    def __getitem__(self, i):
        i = int(i)
        return self.args[0][i]

    def __repr__(self):
        return 'FockState(%r)' % self.args

    def __str__(self):
        return '%s%r%s' % (getattr(self, 'lbracket', ''), self._labels(), getattr(self, 'rbracket', ''))

    def _labels(self):
        return self.args[0]

    def __len__(self):
        return len(self.args[0])

    def _latex(self, printer):
        return '%s%s%s' % (getattr(self, 'lbracket_latex', ''), printer._print(self._labels()), getattr(self, 'rbracket_latex', ''))