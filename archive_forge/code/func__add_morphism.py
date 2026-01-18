from sympy.core import S, Basic, Dict, Symbol, Tuple, sympify
from sympy.core.symbol import Str
from sympy.sets import Set, FiniteSet, EmptySet
from sympy.utilities.iterables import iterable
@staticmethod
def _add_morphism(t, morphism):
    """
        Intelligently adds ``morphism`` to tuple ``t``.

        Explanation
        ===========

        If ``morphism`` is a composite morphism, its components are
        added to the tuple.  If ``morphism`` is an identity, nothing
        is added to the tuple.

        No composability checks are performed.
        """
    if isinstance(morphism, CompositeMorphism):
        return t + morphism.components
    elif isinstance(morphism, IdentityMorphism):
        return t
    else:
        return t + Tuple(morphism)