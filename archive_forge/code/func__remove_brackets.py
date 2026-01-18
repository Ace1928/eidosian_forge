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
def _remove_brackets(self):
    """
        Returns the sorted string without normal order brackets.

        The returned string have the property that no nonzero
        contractions exist.
        """
    subslist = []
    for i in self.iter_q_creators():
        if self[i].is_q_annihilator:
            assume = self[i].state.assumptions0
            if isinstance(self[i].state, Dummy):
                assume.pop('above_fermi', None)
                assume['below_fermi'] = True
                below = Dummy('i', **assume)
                assume.pop('below_fermi', None)
                assume['above_fermi'] = True
                above = Dummy('a', **assume)
                cls = type(self[i])
                split = self[i].__new__(cls, below) * KroneckerDelta(below, self[i].state) + self[i].__new__(cls, above) * KroneckerDelta(above, self[i].state)
                subslist.append((self[i], split))
            else:
                raise SubstitutionOfAmbigousOperatorFailed(self[i])
    if subslist:
        result = NO(self.subs(subslist))
        if isinstance(result, Add):
            return Add(*[term.doit() for term in result.args])
    else:
        return self.args[0]