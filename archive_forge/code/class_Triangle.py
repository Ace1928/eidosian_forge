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
class Triangle(Polygon):
    """
    A polygon with three vertices and three sides.

    Parameters
    ==========

    points : sequence of Points
    keyword: asa, sas, or sss to specify sides/angles of the triangle

    Attributes
    ==========

    vertices
    altitudes
    orthocenter
    circumcenter
    circumradius
    circumcircle
    inradius
    incircle
    exradii
    medians
    medial
    nine_point_circle

    Raises
    ======

    GeometryError
        If the number of vertices is not equal to three, or one of the vertices
        is not a Point, or a valid keyword is not given.

    See Also
    ========

    sympy.geometry.point.Point, Polygon

    Examples
    ========

    >>> from sympy import Triangle, Point
    >>> Triangle(Point(0, 0), Point(4, 0), Point(4, 3))
    Triangle(Point2D(0, 0), Point2D(4, 0), Point2D(4, 3))

    Keywords sss, sas, or asa can be used to give the desired
    side lengths (in order) and interior angles (in degrees) that
    define the triangle:

    >>> Triangle(sss=(3, 4, 5))
    Triangle(Point2D(0, 0), Point2D(3, 0), Point2D(3, 4))
    >>> Triangle(asa=(30, 1, 30))
    Triangle(Point2D(0, 0), Point2D(1, 0), Point2D(1/2, sqrt(3)/6))
    >>> Triangle(sas=(1, 45, 2))
    Triangle(Point2D(0, 0), Point2D(2, 0), Point2D(sqrt(2)/2, sqrt(2)/2))

    """

    def __new__(cls, *args, **kwargs):
        if len(args) != 3:
            if 'sss' in kwargs:
                return _sss(*[simplify(a) for a in kwargs['sss']])
            if 'asa' in kwargs:
                return _asa(*[simplify(a) for a in kwargs['asa']])
            if 'sas' in kwargs:
                return _sas(*[simplify(a) for a in kwargs['sas']])
            msg = 'Triangle instantiates with three points or a valid keyword.'
            raise GeometryError(msg)
        vertices = [Point(a, dim=2, **kwargs) for a in args]
        nodup = []
        for p in vertices:
            if nodup and p == nodup[-1]:
                continue
            nodup.append(p)
        if len(nodup) > 1 and nodup[-1] == nodup[0]:
            nodup.pop()
        i = -3
        while i < len(nodup) - 3 and len(nodup) > 2:
            a, b, c = sorted([nodup[i], nodup[i + 1], nodup[i + 2]], key=default_sort_key)
            if Point.is_collinear(a, b, c):
                nodup[i] = a
                nodup[i + 1] = None
                nodup.pop(i + 1)
            i += 1
        vertices = list(filter(lambda x: x is not None, nodup))
        if len(vertices) == 3:
            return GeometryEntity.__new__(cls, *vertices, **kwargs)
        elif len(vertices) == 2:
            return Segment(*vertices, **kwargs)
        else:
            return Point(*vertices, **kwargs)

    @property
    def vertices(self):
        """The triangle's vertices

        Returns
        =======

        vertices : tuple
            Each element in the tuple is a Point

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Triangle, Point
        >>> t = Triangle(Point(0, 0), Point(4, 0), Point(4, 3))
        >>> t.vertices
        (Point2D(0, 0), Point2D(4, 0), Point2D(4, 3))

        """
        return self.args

    def is_similar(t1, t2):
        """Is another triangle similar to this one.

        Two triangles are similar if one can be uniformly scaled to the other.

        Parameters
        ==========

        other: Triangle

        Returns
        =======

        is_similar : boolean

        See Also
        ========

        sympy.geometry.entity.GeometryEntity.is_similar

        Examples
        ========

        >>> from sympy import Triangle, Point
        >>> t1 = Triangle(Point(0, 0), Point(4, 0), Point(4, 3))
        >>> t2 = Triangle(Point(0, 0), Point(-4, 0), Point(-4, -3))
        >>> t1.is_similar(t2)
        True

        >>> t2 = Triangle(Point(0, 0), Point(-4, 0), Point(-4, -4))
        >>> t1.is_similar(t2)
        False

        """
        if not isinstance(t2, Polygon):
            return False
        s1_1, s1_2, s1_3 = [side.length for side in t1.sides]
        s2 = [side.length for side in t2.sides]

        def _are_similar(u1, u2, u3, v1, v2, v3):
            e1 = simplify(u1 / v1)
            e2 = simplify(u2 / v2)
            e3 = simplify(u3 / v3)
            return bool(e1 == e2) and bool(e2 == e3)
        return _are_similar(s1_1, s1_2, s1_3, *s2) or _are_similar(s1_1, s1_3, s1_2, *s2) or _are_similar(s1_2, s1_1, s1_3, *s2) or _are_similar(s1_2, s1_3, s1_1, *s2) or _are_similar(s1_3, s1_1, s1_2, *s2) or _are_similar(s1_3, s1_2, s1_1, *s2)

    def is_equilateral(self):
        """Are all the sides the same length?

        Returns
        =======

        is_equilateral : boolean

        See Also
        ========

        sympy.geometry.entity.GeometryEntity.is_similar, RegularPolygon
        is_isosceles, is_right, is_scalene

        Examples
        ========

        >>> from sympy import Triangle, Point
        >>> t1 = Triangle(Point(0, 0), Point(4, 0), Point(4, 3))
        >>> t1.is_equilateral()
        False

        >>> from sympy import sqrt
        >>> t2 = Triangle(Point(0, 0), Point(10, 0), Point(5, 5*sqrt(3)))
        >>> t2.is_equilateral()
        True

        """
        return not has_variety((s.length for s in self.sides))

    def is_isosceles(self):
        """Are two or more of the sides the same length?

        Returns
        =======

        is_isosceles : boolean

        See Also
        ========

        is_equilateral, is_right, is_scalene

        Examples
        ========

        >>> from sympy import Triangle, Point
        >>> t1 = Triangle(Point(0, 0), Point(4, 0), Point(2, 4))
        >>> t1.is_isosceles()
        True

        """
        return has_dups((s.length for s in self.sides))

    def is_scalene(self):
        """Are all the sides of the triangle of different lengths?

        Returns
        =======

        is_scalene : boolean

        See Also
        ========

        is_equilateral, is_isosceles, is_right

        Examples
        ========

        >>> from sympy import Triangle, Point
        >>> t1 = Triangle(Point(0, 0), Point(4, 0), Point(1, 4))
        >>> t1.is_scalene()
        True

        """
        return not has_dups((s.length for s in self.sides))

    def is_right(self):
        """Is the triangle right-angled.

        Returns
        =======

        is_right : boolean

        See Also
        ========

        sympy.geometry.line.LinearEntity.is_perpendicular
        is_equilateral, is_isosceles, is_scalene

        Examples
        ========

        >>> from sympy import Triangle, Point
        >>> t1 = Triangle(Point(0, 0), Point(4, 0), Point(4, 3))
        >>> t1.is_right()
        True

        """
        s = self.sides
        return Segment.is_perpendicular(s[0], s[1]) or Segment.is_perpendicular(s[1], s[2]) or Segment.is_perpendicular(s[0], s[2])

    @property
    def altitudes(self):
        """The altitudes of the triangle.

        An altitude of a triangle is a segment through a vertex,
        perpendicular to the opposite side, with length being the
        height of the vertex measured from the line containing the side.

        Returns
        =======

        altitudes : dict
            The dictionary consists of keys which are vertices and values
            which are Segments.

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.line.Segment.length

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.altitudes[p1]
        Segment2D(Point2D(0, 0), Point2D(1/2, 1/2))

        """
        s = self.sides
        v = self.vertices
        return {v[0]: s[1].perpendicular_segment(v[0]), v[1]: s[2].perpendicular_segment(v[1]), v[2]: s[0].perpendicular_segment(v[2])}

    @property
    def orthocenter(self):
        """The orthocenter of the triangle.

        The orthocenter is the intersection of the altitudes of a triangle.
        It may lie inside, outside or on the triangle.

        Returns
        =======

        orthocenter : Point

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.orthocenter
        Point2D(0, 0)

        """
        a = self.altitudes
        v = self.vertices
        return Line(a[v[0]]).intersection(Line(a[v[1]]))[0]

    @property
    def circumcenter(self):
        """The circumcenter of the triangle

        The circumcenter is the center of the circumcircle.

        Returns
        =======

        circumcenter : Point

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.circumcenter
        Point2D(1/2, 1/2)
        """
        a, b, c = [x.perpendicular_bisector() for x in self.sides]
        return a.intersection(b)[0]

    @property
    def circumradius(self):
        """The radius of the circumcircle of the triangle.

        Returns
        =======

        circumradius : number of Basic instance

        See Also
        ========

        sympy.geometry.ellipse.Circle.radius

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy import Point, Triangle
        >>> a = Symbol('a')
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, a)
        >>> t = Triangle(p1, p2, p3)
        >>> t.circumradius
        sqrt(a**2/4 + 1/4)
        """
        return Point.distance(self.circumcenter, self.vertices[0])

    @property
    def circumcircle(self):
        """The circle which passes through the three vertices of the triangle.

        Returns
        =======

        circumcircle : Circle

        See Also
        ========

        sympy.geometry.ellipse.Circle

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.circumcircle
        Circle(Point2D(1/2, 1/2), sqrt(2)/2)

        """
        return Circle(self.circumcenter, self.circumradius)

    def bisectors(self):
        """The angle bisectors of the triangle.

        An angle bisector of a triangle is a straight line through a vertex
        which cuts the corresponding angle in half.

        Returns
        =======

        bisectors : dict
            Each key is a vertex (Point) and each value is the corresponding
            bisector (Segment).

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.line.Segment

        Examples
        ========

        >>> from sympy import Point, Triangle, Segment
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> from sympy import sqrt
        >>> t.bisectors()[p2] == Segment(Point(1, 0), Point(0, sqrt(2) - 1))
        True

        """
        s = [Line(l) for l in self.sides]
        v = self.vertices
        c = self.incenter
        l1 = Segment(v[0], Line(v[0], c).intersection(s[1])[0])
        l2 = Segment(v[1], Line(v[1], c).intersection(s[2])[0])
        l3 = Segment(v[2], Line(v[2], c).intersection(s[0])[0])
        return {v[0]: l1, v[1]: l2, v[2]: l3}

    @property
    def incenter(self):
        """The center of the incircle.

        The incircle is the circle which lies inside the triangle and touches
        all three sides.

        Returns
        =======

        incenter : Point

        See Also
        ========

        incircle, sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.incenter
        Point2D(1 - sqrt(2)/2, 1 - sqrt(2)/2)

        """
        s = self.sides
        l = Matrix([s[i].length for i in [1, 2, 0]])
        p = sum(l)
        v = self.vertices
        x = simplify(l.dot(Matrix([vi.x for vi in v])) / p)
        y = simplify(l.dot(Matrix([vi.y for vi in v])) / p)
        return Point(x, y)

    @property
    def inradius(self):
        """The radius of the incircle.

        Returns
        =======

        inradius : number of Basic instance

        See Also
        ========

        incircle, sympy.geometry.ellipse.Circle.radius

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(4, 0), Point(0, 3)
        >>> t = Triangle(p1, p2, p3)
        >>> t.inradius
        1

        """
        return simplify(2 * self.area / self.perimeter)

    @property
    def incircle(self):
        """The incircle of the triangle.

        The incircle is the circle which lies inside the triangle and touches
        all three sides.

        Returns
        =======

        incircle : Circle

        See Also
        ========

        sympy.geometry.ellipse.Circle

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(2, 0), Point(0, 2)
        >>> t = Triangle(p1, p2, p3)
        >>> t.incircle
        Circle(Point2D(2 - sqrt(2), 2 - sqrt(2)), 2 - sqrt(2))

        """
        return Circle(self.incenter, self.inradius)

    @property
    def exradii(self):
        """The radius of excircles of a triangle.

        An excircle of the triangle is a circle lying outside the triangle,
        tangent to one of its sides and tangent to the extensions of the
        other two.

        Returns
        =======

        exradii : dict

        See Also
        ========

        sympy.geometry.polygon.Triangle.inradius

        Examples
        ========

        The exradius touches the side of the triangle to which it is keyed, e.g.
        the exradius touching side 2 is:

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(6, 0), Point(0, 2)
        >>> t = Triangle(p1, p2, p3)
        >>> t.exradii[t.sides[2]]
        -2 + sqrt(10)

        References
        ==========

        .. [1] https://mathworld.wolfram.com/Exradius.html
        .. [2] https://mathworld.wolfram.com/Excircles.html

        """
        side = self.sides
        a = side[0].length
        b = side[1].length
        c = side[2].length
        s = (a + b + c) / 2
        area = self.area
        exradii = {self.sides[0]: simplify(area / (s - a)), self.sides[1]: simplify(area / (s - b)), self.sides[2]: simplify(area / (s - c))}
        return exradii

    @property
    def excenters(self):
        """Excenters of the triangle.

        An excenter is the center of a circle that is tangent to a side of the
        triangle and the extensions of the other two sides.

        Returns
        =======

        excenters : dict


        Examples
        ========

        The excenters are keyed to the side of the triangle to which their corresponding
        excircle is tangent: The center is keyed, e.g. the excenter of a circle touching
        side 0 is:

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(6, 0), Point(0, 2)
        >>> t = Triangle(p1, p2, p3)
        >>> t.excenters[t.sides[0]]
        Point2D(12*sqrt(10), 2/3 + sqrt(10)/3)

        See Also
        ========

        sympy.geometry.polygon.Triangle.exradii

        References
        ==========

        .. [1] https://mathworld.wolfram.com/Excircles.html

        """
        s = self.sides
        v = self.vertices
        a = s[0].length
        b = s[1].length
        c = s[2].length
        x = [v[0].x, v[1].x, v[2].x]
        y = [v[0].y, v[1].y, v[2].y]
        exc_coords = {'x1': simplify(-a * x[0] + b * x[1] + c * x[2] / (-a + b + c)), 'x2': simplify(a * x[0] - b * x[1] + c * x[2] / (a - b + c)), 'x3': simplify(a * x[0] + b * x[1] - c * x[2] / (a + b - c)), 'y1': simplify(-a * y[0] + b * y[1] + c * y[2] / (-a + b + c)), 'y2': simplify(a * y[0] - b * y[1] + c * y[2] / (a - b + c)), 'y3': simplify(a * y[0] + b * y[1] - c * y[2] / (a + b - c))}
        excenters = {s[0]: Point(exc_coords['x1'], exc_coords['y1']), s[1]: Point(exc_coords['x2'], exc_coords['y2']), s[2]: Point(exc_coords['x3'], exc_coords['y3'])}
        return excenters

    @property
    def medians(self):
        """The medians of the triangle.

        A median of a triangle is a straight line through a vertex and the
        midpoint of the opposite side, and divides the triangle into two
        equal areas.

        Returns
        =======

        medians : dict
            Each key is a vertex (Point) and each value is the median (Segment)
            at that point.

        See Also
        ========

        sympy.geometry.point.Point.midpoint, sympy.geometry.line.Segment.midpoint

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.medians[p1]
        Segment2D(Point2D(0, 0), Point2D(1/2, 1/2))

        """
        s = self.sides
        v = self.vertices
        return {v[0]: Segment(v[0], s[1].midpoint), v[1]: Segment(v[1], s[2].midpoint), v[2]: Segment(v[2], s[0].midpoint)}

    @property
    def medial(self):
        """The medial triangle of the triangle.

        The triangle which is formed from the midpoints of the three sides.

        Returns
        =======

        medial : Triangle

        See Also
        ========

        sympy.geometry.line.Segment.midpoint

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.medial
        Triangle(Point2D(1/2, 0), Point2D(1/2, 1/2), Point2D(0, 1/2))

        """
        s = self.sides
        return Triangle(s[0].midpoint, s[1].midpoint, s[2].midpoint)

    @property
    def nine_point_circle(self):
        """The nine-point circle of the triangle.

        Nine-point circle is the circumcircle of the medial triangle, which
        passes through the feet of altitudes and the middle points of segments
        connecting the vertices and the orthocenter.

        Returns
        =======

        nine_point_circle : Circle

        See also
        ========

        sympy.geometry.line.Segment.midpoint
        sympy.geometry.polygon.Triangle.medial
        sympy.geometry.polygon.Triangle.orthocenter

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.nine_point_circle
        Circle(Point2D(1/4, 1/4), sqrt(2)/4)

        """
        return Circle(*self.medial.vertices)

    @property
    def eulerline(self):
        """The Euler line of the triangle.

        The line which passes through circumcenter, centroid and orthocenter.

        Returns
        =======

        eulerline : Line (or Point for equilateral triangles in which case all
                    centers coincide)

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.eulerline
        Line2D(Point2D(0, 0), Point2D(1/2, 1/2))

        """
        if self.is_equilateral():
            return self.orthocenter
        return Line(self.orthocenter, self.circumcenter)