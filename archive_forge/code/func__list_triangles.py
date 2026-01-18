from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
@staticmethod
def _list_triangles(edges):
    """
        Builds the set of triangles formed by the supplied edges.  The
        triangles are arbitrary and need not be commutative.  A
        triangle is a set that contains all three of its sides.
        """
    triangles = set()
    for w in edges:
        for v in edges:
            wv = DiagramGrid._juxtapose_edges(w, v)
            if wv and wv in edges:
                triangles.add(frozenset([w, v, wv]))
    return triangles