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
class Segment2D(LinearEntity2D, Segment):
    """A line segment in 2D space.

    Parameters
    ==========

    p1 : Point
    p2 : Point

    Attributes
    ==========

    length : number or SymPy expression
    midpoint : Point

    See Also
    ========

    sympy.geometry.point.Point, Line

    Examples
    ========

    >>> from sympy import Point, Segment
    >>> Segment((1, 0), (1, 1)) # tuples are interpreted as pts
    Segment2D(Point2D(1, 0), Point2D(1, 1))
    >>> s = Segment(Point(4, 3), Point(1, 1)); s
    Segment2D(Point2D(4, 3), Point2D(1, 1))
    >>> s.points
    (Point2D(4, 3), Point2D(1, 1))
    >>> s.slope
    2/3
    >>> s.length
    sqrt(13)
    >>> s.midpoint
    Point2D(5/2, 2)

    """

    def __new__(cls, p1, p2, **kwargs):
        p1 = Point(p1, dim=2)
        p2 = Point(p2, dim=2)
        if p1 == p2:
            return p1
        return LinearEntity2D.__new__(cls, p1, p2, **kwargs)

    def _svg(self, scale_factor=1.0, fill_color='#66cc99'):
        """Returns SVG path element for the LinearEntity.

        Parameters
        ==========

        scale_factor : float
            Multiplication factor for the SVG stroke-width.  Default is 1.
        fill_color : str, optional
            Hex string for fill color. Default is "#66cc99".
        """
        verts = (N(self.p1), N(self.p2))
        coords = ['{},{}'.format(p.x, p.y) for p in verts]
        path = 'M {} L {}'.format(coords[0], ' L '.join(coords[1:]))
        return '<path fill-rule="evenodd" fill="{2}" stroke="#555555" stroke-width="{0}" opacity="0.6" d="{1}" />'.format(2.0 * scale_factor, path, fill_color)