from collections import deque
from math import sqrt as _sqrt
from .entity import GeometryEntity
from .exceptions import GeometryError
from .point import Point, Point2D, Point3D
from sympy.core.containers import OrderedSet
from sympy.core.exprtools import factor_terms
from sympy.core.function import Function, expand_mul
from sympy.core.sorting import ordered
from sympy.core.symbol import Symbol
from sympy.core.singleton import S
from sympy.polys.polytools import cancel
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.utilities.iterables import is_sequence
def _orientation(p, q, r):
    """Return positive if p-q-r are clockwise, neg if ccw, zero if
        collinear."""
    return (q.y - p.y) * (r.x - p.x) - (q.x - p.x) * (r.y - p.y)