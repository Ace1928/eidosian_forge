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
class FockStateBra(FockState):
    """
    Representation of a bra.
    """
    lbracket = '<'
    rbracket = '|'
    lbracket_latex = '\\left\\langle'
    rbracket_latex = '\\right|'

    def __mul__(self, other):
        if isinstance(other, FockStateKet):
            return InnerProduct(self, other)
        else:
            return Expr.__mul__(self, other)