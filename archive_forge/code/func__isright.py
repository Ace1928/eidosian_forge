from sympy.core import Expr, S, oo, pi, sympify
from sympy.core.evalf import N
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import _symbol, Dummy, Symbol
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import cos, sin, tan
from .ellipse import Circle
from .entity import GeometryEntity, GeometrySet
from .exceptions import GeometryError
from .line import Line, Segment, Ray
from .point import Point
from sympy.logic import And
from sympy.matrices import Matrix
from sympy.simplify.simplify import simplify
from sympy.solvers.solvers import solve
from sympy.utilities.iterables import has_dups, has_variety, uniq, rotate_left, least_rotation
from sympy.utilities.misc import as_int, func_name
from mpmath.libmp.libmpf import prec_to_dps
import warnings
@staticmethod
def _isright(a, b, c):
    """Return True/False for cw/ccw orientation.

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> a, b, c = [Point(i) for i in [(0, 0), (1, 1), (1, 0)]]
        >>> Polygon._isright(a, b, c)
        True
        >>> Polygon._isright(a, c, b)
        False
        """
    ba = b - a
    ca = c - a
    t_area = simplify(ba.x * ca.y - ca.x * ba.y)
    res = t_area.is_nonpositive
    if res is None:
        raise ValueError("Can't determine orientation")
    return res