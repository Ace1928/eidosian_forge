from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.combinatorics.free_groups import (FreeGroup, FreeGroupElement,
from sympy.combinatorics.rewritingsystem import RewritingSystem
from sympy.combinatorics.coset_table import (CosetTable,
from sympy.combinatorics import PermutationGroup
from sympy.matrices.normalforms import invariant_factors
from sympy.matrices import Matrix
from sympy.polys.polytools import gcd
from sympy.printing.defaults import DefaultPrinting
from sympy.utilities import public
from sympy.utilities.magic import pollute
from itertools import product
def _word_break(w):
    if len(w) == 0:
        return True
    i = 0
    while i < len(w):
        i += 1
        prefix = w.subword(0, i)
        if not _is_subword(prefix):
            continue
        rest = w.subword(i, len(w))
        if rest not in known:
            known[rest] = _word_break(rest)
        if known[rest]:
            return True
    return False