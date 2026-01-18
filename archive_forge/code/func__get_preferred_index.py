from math import prod
from sympy.core import S, Integer
from sympy.core.function import Function
from sympy.core.logic import fuzzy_not
from sympy.core.relational import Ne
from sympy.core.sorting import default_sort_key
from sympy.external.gmpy import SYMPY_INTS
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.piecewise import Piecewise
from sympy.utilities.iterables import has_dups
def _get_preferred_index(self):
    """
        Returns the index which is preferred to keep in the final expression.

        The preferred index is the index with more information regarding fermi
        level. If indices contain the same information, index 0 is returned.

        """
    if not self.is_above_fermi:
        if self.args[0].assumptions0.get('below_fermi'):
            return 0
        else:
            return 1
    elif not self.is_below_fermi:
        if self.args[0].assumptions0.get('above_fermi'):
            return 0
        else:
            return 1
    else:
        return 0