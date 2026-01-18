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
class FockStateFermionKet(FermionState, FockStateKet):
    """
    Many-particle Fock state with a sequence of occupied orbits.

    Explanation
    ===========

    Each state can only have one particle, so we choose to store a list of
    occupied orbits rather than a tuple with occupation numbers (zeros and ones).

    states below fermi level are holes, and are represented by negative labels
    in the occupation list.

    For symbolic state labels, the fermi_level caps the number of allowed hole-
    states.

    Examples
    ========

    >>> from sympy.physics.secondquant import FKet
    >>> FKet([1, 2])
    FockStateFermionKet((1, 2))
    """

    def _dagger_(self):
        return FockStateFermionBra(*self.args)