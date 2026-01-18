from sympy.core.containers import Tuple
from sympy.core.evalf import N
from sympy.core.expr import Expr
from sympy.core.numbers import Rational, oo, Float
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import _symbol, Dummy, uniquely_named_symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (_pi_coeff, acos, tan, atan2)
from .entity import GeometryEntity, GeometrySet
from .exceptions import GeometryError
from .point import Point, Point3D
from .util import find, intersection
from sympy.logic.boolalg import And
from sympy.matrices import Matrix
from sympy.sets.sets import Intersection
from sympy.simplify.simplify import simplify
from sympy.solvers.solvers import solve
from sympy.solvers.solveset import linear_coeffs
from sympy.utilities.misc import Undecidable, filldedent
import random
class Segment3D(LinearEntity3D, Segment):
    """A line segment in a 3D space.

    Parameters
    ==========

    p1 : Point3D
    p2 : Point3D

    Attributes
    ==========

    length : number or SymPy expression
    midpoint : Point3D

    See Also
    ========

    sympy.geometry.point.Point3D, Line3D

    Examples
    ========

    >>> from sympy import Point3D, Segment3D
    >>> Segment3D((1, 0, 0), (1, 1, 1)) # tuples are interpreted as pts
    Segment3D(Point3D(1, 0, 0), Point3D(1, 1, 1))
    >>> s = Segment3D(Point3D(4, 3, 9), Point3D(1, 1, 7)); s
    Segment3D(Point3D(4, 3, 9), Point3D(1, 1, 7))
    >>> s.points
    (Point3D(4, 3, 9), Point3D(1, 1, 7))
    >>> s.length
    sqrt(17)
    >>> s.midpoint
    Point3D(5/2, 2, 8)

    """

    def __new__(cls, p1, p2, **kwargs):
        p1 = Point(p1, dim=3)
        p2 = Point(p2, dim=3)
        if p1 == p2:
            return p1
        return LinearEntity3D.__new__(cls, p1, p2, **kwargs)