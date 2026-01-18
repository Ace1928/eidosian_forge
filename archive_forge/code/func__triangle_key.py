from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
@staticmethod
def _triangle_key(tri, triangle_sizes):
    """
        Returns a key for the supplied triangle.  It should be the
        same independently of the hash randomisation.
        """
    objects = sorted(DiagramGrid._triangle_objects(tri), key=default_sort_key)
    return (triangle_sizes[tri], default_sort_key(objects))