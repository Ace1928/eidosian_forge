from __future__ import annotations
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.evalf import EvalfMixin, N
from sympy.core.numbers import oo
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import cos, sin, atan
from sympy.matrices import eye
from sympy.multipledispatch import dispatch
from sympy.printing import sstr
from sympy.sets import Set, Union, FiniteSet
from sympy.sets.handlers.intersection import intersection_sets
from sympy.sets.handlers.union import union_sets
from sympy.solvers.solvers import solve
from sympy.utilities.misc import func_name
from sympy.utilities.iterables import is_sequence
class GeometrySet(GeometryEntity, Set):
    """Parent class of all GeometryEntity that are also Sets
    (compatible with sympy.sets)
    """
    __slots__ = ()

    def _contains(self, other):
        """sympy.sets uses the _contains method, so include it for compatibility."""
        if isinstance(other, Set) and other.is_FiniteSet:
            return all((self.__contains__(i) for i in other))
        return self.__contains__(other)