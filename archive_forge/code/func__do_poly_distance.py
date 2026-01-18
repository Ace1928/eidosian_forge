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
def _do_poly_distance(self, e2):
    """
        Calculates the least distance between the exteriors of two
        convex polygons e1 and e2. Does not check for the convexity
        of the polygons as this is checked by Polygon.distance.

        Notes
        =====

            - Prints a warning if the two polygons possibly intersect as the return
              value will not be valid in such a case. For a more through test of
              intersection use intersection().

        See Also
        ========

        sympy.geometry.point.Point.distance

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> square = Polygon(Point(0, 0), Point(0, 1), Point(1, 1), Point(1, 0))
        >>> triangle = Polygon(Point(1, 2), Point(2, 2), Point(2, 1))
        >>> square._do_poly_distance(triangle)
        sqrt(2)/2

        Description of method used
        ==========================

        Method:
        [1] https://web.archive.org/web/20150509035744/http://cgm.cs.mcgill.ca/~orm/mind2p.html
        Uses rotating calipers:
        [2] https://en.wikipedia.org/wiki/Rotating_calipers
        and antipodal points:
        [3] https://en.wikipedia.org/wiki/Antipodal_point
        """
    e1 = self
    'Tests for a possible intersection between the polygons and outputs a warning'
    e1_center = e1.centroid
    e2_center = e2.centroid
    e1_max_radius = S.Zero
    e2_max_radius = S.Zero
    for vertex in e1.vertices:
        r = Point.distance(e1_center, vertex)
        if e1_max_radius < r:
            e1_max_radius = r
    for vertex in e2.vertices:
        r = Point.distance(e2_center, vertex)
        if e2_max_radius < r:
            e2_max_radius = r
    center_dist = Point.distance(e1_center, e2_center)
    if center_dist <= e1_max_radius + e2_max_radius:
        warnings.warn('Polygons may intersect producing erroneous output', stacklevel=3)
    '\n        Find the upper rightmost vertex of e1 and the lowest leftmost vertex of e2\n        '
    e1_ymax = Point(0, -oo)
    e2_ymin = Point(0, oo)
    for vertex in e1.vertices:
        if vertex.y > e1_ymax.y or (vertex.y == e1_ymax.y and vertex.x > e1_ymax.x):
            e1_ymax = vertex
    for vertex in e2.vertices:
        if vertex.y < e2_ymin.y or (vertex.y == e2_ymin.y and vertex.x < e2_ymin.x):
            e2_ymin = vertex
    min_dist = Point.distance(e1_ymax, e2_ymin)
    '\n        Produce a dictionary with vertices of e1 as the keys and, for each vertex, the points\n        to which the vertex is connected as its value. The same is then done for e2.\n        '
    e1_connections = {}
    e2_connections = {}
    for side in e1.sides:
        if side.p1 in e1_connections:
            e1_connections[side.p1].append(side.p2)
        else:
            e1_connections[side.p1] = [side.p2]
        if side.p2 in e1_connections:
            e1_connections[side.p2].append(side.p1)
        else:
            e1_connections[side.p2] = [side.p1]
    for side in e2.sides:
        if side.p1 in e2_connections:
            e2_connections[side.p1].append(side.p2)
        else:
            e2_connections[side.p1] = [side.p2]
        if side.p2 in e2_connections:
            e2_connections[side.p2].append(side.p1)
        else:
            e2_connections[side.p2] = [side.p1]
    e1_current = e1_ymax
    e2_current = e2_ymin
    support_line = Line(Point(S.Zero, S.Zero), Point(S.One, S.Zero))
    '\n        Determine which point in e1 and e2 will be selected after e2_ymin and e1_ymax,\n        this information combined with the above produced dictionaries determines the\n        path that will be taken around the polygons\n        '
    point1 = e1_connections[e1_ymax][0]
    point2 = e1_connections[e1_ymax][1]
    angle1 = support_line.angle_between(Line(e1_ymax, point1))
    angle2 = support_line.angle_between(Line(e1_ymax, point2))
    if angle1 < angle2:
        e1_next = point1
    elif angle2 < angle1:
        e1_next = point2
    elif Point.distance(e1_ymax, point1) > Point.distance(e1_ymax, point2):
        e1_next = point2
    else:
        e1_next = point1
    point1 = e2_connections[e2_ymin][0]
    point2 = e2_connections[e2_ymin][1]
    angle1 = support_line.angle_between(Line(e2_ymin, point1))
    angle2 = support_line.angle_between(Line(e2_ymin, point2))
    if angle1 > angle2:
        e2_next = point1
    elif angle2 > angle1:
        e2_next = point2
    elif Point.distance(e2_ymin, point1) > Point.distance(e2_ymin, point2):
        e2_next = point2
    else:
        e2_next = point1
    '\n        Loop which determines the distance between anti-podal pairs and updates the\n        minimum distance accordingly. It repeats until it reaches the starting position.\n        '
    while True:
        e1_angle = support_line.angle_between(Line(e1_current, e1_next))
        e2_angle = pi - support_line.angle_between(Line(e2_current, e2_next))
        if (e1_angle < e2_angle) is True:
            support_line = Line(e1_current, e1_next)
            e1_segment = Segment(e1_current, e1_next)
            min_dist_current = e1_segment.distance(e2_current)
            if min_dist_current.evalf() < min_dist.evalf():
                min_dist = min_dist_current
            if e1_connections[e1_next][0] != e1_current:
                e1_current = e1_next
                e1_next = e1_connections[e1_next][0]
            else:
                e1_current = e1_next
                e1_next = e1_connections[e1_next][1]
        elif (e1_angle > e2_angle) is True:
            support_line = Line(e2_next, e2_current)
            e2_segment = Segment(e2_current, e2_next)
            min_dist_current = e2_segment.distance(e1_current)
            if min_dist_current.evalf() < min_dist.evalf():
                min_dist = min_dist_current
            if e2_connections[e2_next][0] != e2_current:
                e2_current = e2_next
                e2_next = e2_connections[e2_next][0]
            else:
                e2_current = e2_next
                e2_next = e2_connections[e2_next][1]
        else:
            support_line = Line(e1_current, e1_next)
            e1_segment = Segment(e1_current, e1_next)
            e2_segment = Segment(e2_current, e2_next)
            min1 = e1_segment.distance(e2_next)
            min2 = e2_segment.distance(e1_next)
            min_dist_current = min(min1, min2)
            if min_dist_current.evalf() < min_dist.evalf():
                min_dist = min_dist_current
            if e1_connections[e1_next][0] != e1_current:
                e1_current = e1_next
                e1_next = e1_connections[e1_next][0]
            else:
                e1_current = e1_next
                e1_next = e1_connections[e1_next][1]
            if e2_connections[e2_next][0] != e2_current:
                e2_current = e2_next
                e2_next = e2_connections[e2_next][0]
            else:
                e2_current = e2_next
                e2_next = e2_connections[e2_next][1]
        if e1_current == e1_ymax and e2_current == e2_ymin:
            break
    return min_dist