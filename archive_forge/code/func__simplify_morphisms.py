from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
@staticmethod
def _simplify_morphisms(morphisms):
    """
        Given a dictionary mapping morphisms to their properties,
        returns a new dictionary in which there are no morphisms which
        do not have properties, and which are compositions of other
        morphisms included in the dictionary.  Identities are dropped
        as well.
        """
    newmorphisms = {}
    for morphism, props in morphisms.items():
        if isinstance(morphism, CompositeMorphism) and (not props):
            continue
        elif isinstance(morphism, IdentityMorphism):
            continue
        else:
            newmorphisms[morphism] = props
    return newmorphisms