from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
@staticmethod
def _pick_root_edge(tri, skeleton):
    """
        For a given triangle always picks the same root edge.  The
        root edge is the edge that will be placed first on the grid.
        """
    candidates = [sorted(e, key=default_sort_key) for e in tri if skeleton[e]]
    sorted_candidates = sorted(candidates, key=default_sort_key)
    return tuple(sorted(sorted_candidates[0], key=default_sort_key))