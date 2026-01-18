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
def _remove_orbit(self, i):
    """
        Removes particle/fills hole in orbit i. No input tests performed here.
        """
    new_occs = list(self.args[0])
    pos = new_occs.index(i)
    del new_occs[pos]
    if pos % 2:
        return S.NegativeOne * self.__class__(new_occs, self.fermi_level)
    else:
        return self.__class__(new_occs, self.fermi_level)