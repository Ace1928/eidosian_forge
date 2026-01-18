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
class LinearEntity(GeometrySet):
    """A base class for all linear entities (Line, Ray and Segment)
    in n-dimensional Euclidean space.

    Attributes
    ==========

    ambient_dimension
    direction
    length
    p1
    p2
    points

    Notes
    =====

    This is an abstract class and is not meant to be instantiated.

    See Also
    ========

    sympy.geometry.entity.GeometryEntity

    """

    def __new__(cls, p1, p2=None, **kwargs):
        p1, p2 = Point._normalize_dimension(p1, p2)
        if p1 == p2:
            raise ValueError('%s.__new__ requires two unique Points.' % cls.__name__)
        if len(p1) != len(p2):
            raise ValueError('%s.__new__ requires two Points of equal dimension.' % cls.__name__)
        return GeometryEntity.__new__(cls, p1, p2, **kwargs)

    def __contains__(self, other):
        """Return a definitive answer or else raise an error if it cannot
        be determined that other is on the boundaries of self."""
        result = self.contains(other)
        if result is not None:
            return result
        else:
            raise Undecidable("Cannot decide whether '%s' contains '%s'" % (self, other))

    def _span_test(self, other):
        """Test whether the point `other` lies in the positive span of `self`.
        A point x is 'in front' of a point y if x.dot(y) >= 0.  Return
        -1 if `other` is behind `self.p1`, 0 if `other` is `self.p1` and
        and 1 if `other` is in front of `self.p1`."""
        if self.p1 == other:
            return 0
        rel_pos = other - self.p1
        d = self.direction
        if d.dot(rel_pos) > 0:
            return 1
        return -1

    @property
    def ambient_dimension(self):
        """A property method that returns the dimension of LinearEntity
        object.

        Parameters
        ==========

        p1 : LinearEntity

        Returns
        =======

        dimension : integer

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(0, 0), Point(1, 1)
        >>> l1 = Line(p1, p2)
        >>> l1.ambient_dimension
        2

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(0, 0, 0), Point(1, 1, 1)
        >>> l1 = Line(p1, p2)
        >>> l1.ambient_dimension
        3

        """
        return len(self.p1)

    def angle_between(l1, l2):
        """Return the non-reflex angle formed by rays emanating from
        the origin with directions the same as the direction vectors
        of the linear entities.

        Parameters
        ==========

        l1 : LinearEntity
        l2 : LinearEntity

        Returns
        =======

        angle : angle in radians

        Notes
        =====

        From the dot product of vectors v1 and v2 it is known that:

            ``dot(v1, v2) = |v1|*|v2|*cos(A)``

        where A is the angle formed between the two vectors. We can
        get the directional vectors of the two lines and readily
        find the angle between the two using the above formula.

        See Also
        ========

        is_perpendicular, Ray2D.closing_angle

        Examples
        ========

        >>> from sympy import Line
        >>> e = Line((0, 0), (1, 0))
        >>> ne = Line((0, 0), (1, 1))
        >>> sw = Line((1, 1), (0, 0))
        >>> ne.angle_between(e)
        pi/4
        >>> sw.angle_between(e)
        3*pi/4

        To obtain the non-obtuse angle at the intersection of lines, use
        the ``smallest_angle_between`` method:

        >>> sw.smallest_angle_between(e)
        pi/4

        >>> from sympy import Point3D, Line3D
        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(1, 1, 1), Point3D(-1, 2, 0)
        >>> l1, l2 = Line3D(p1, p2), Line3D(p2, p3)
        >>> l1.angle_between(l2)
        acos(-sqrt(2)/3)
        >>> l1.smallest_angle_between(l2)
        acos(sqrt(2)/3)
        """
        if not isinstance(l1, LinearEntity) and (not isinstance(l2, LinearEntity)):
            raise TypeError('Must pass only LinearEntity objects')
        v1, v2 = (l1.direction, l2.direction)
        return acos(v1.dot(v2) / (abs(v1) * abs(v2)))

    def smallest_angle_between(l1, l2):
        """Return the smallest angle formed at the intersection of the
        lines containing the linear entities.

        Parameters
        ==========

        l1 : LinearEntity
        l2 : LinearEntity

        Returns
        =======

        angle : angle in radians

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2, p3 = Point(0, 0), Point(0, 4), Point(2, -2)
        >>> l1, l2 = Line(p1, p2), Line(p1, p3)
        >>> l1.smallest_angle_between(l2)
        pi/4

        See Also
        ========

        angle_between, is_perpendicular, Ray2D.closing_angle
        """
        if not isinstance(l1, LinearEntity) and (not isinstance(l2, LinearEntity)):
            raise TypeError('Must pass only LinearEntity objects')
        v1, v2 = (l1.direction, l2.direction)
        return acos(abs(v1.dot(v2)) / (abs(v1) * abs(v2)))

    def arbitrary_point(self, parameter='t'):
        """A parameterized point on the Line.

        Parameters
        ==========

        parameter : str, optional
            The name of the parameter which will be used for the parametric
            point. The default value is 't'. When this parameter is 0, the
            first point used to define the line will be returned, and when
            it is 1 the second point will be returned.

        Returns
        =======

        point : Point

        Raises
        ======

        ValueError
            When ``parameter`` already appears in the Line's definition.

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(1, 0), Point(5, 3)
        >>> l1 = Line(p1, p2)
        >>> l1.arbitrary_point()
        Point2D(4*t + 1, 3*t)
        >>> from sympy import Point3D, Line3D
        >>> p1, p2 = Point3D(1, 0, 0), Point3D(5, 3, 1)
        >>> l1 = Line3D(p1, p2)
        >>> l1.arbitrary_point()
        Point3D(4*t + 1, 3*t, t)

        """
        t = _symbol(parameter, real=True)
        if t.name in (f.name for f in self.free_symbols):
            raise ValueError(filldedent('\n                Symbol %s already appears in object\n                and cannot be used as a parameter.\n                ' % t.name))
        return self.p1 + (self.p2 - self.p1) * t

    @staticmethod
    def are_concurrent(*lines):
        """Is a sequence of linear entities concurrent?

        Two or more linear entities are concurrent if they all
        intersect at a single point.

        Parameters
        ==========

        lines
            A sequence of linear entities.

        Returns
        =======

        True : if the set of linear entities intersect in one point
        False : otherwise.

        See Also
        ========

        sympy.geometry.util.intersection

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(0, 0), Point(3, 5)
        >>> p3, p4 = Point(-2, -2), Point(0, 2)
        >>> l1, l2, l3 = Line(p1, p2), Line(p1, p3), Line(p1, p4)
        >>> Line.are_concurrent(l1, l2, l3)
        True
        >>> l4 = Line(p2, p3)
        >>> Line.are_concurrent(l2, l3, l4)
        False
        >>> from sympy import Point3D, Line3D
        >>> p1, p2 = Point3D(0, 0, 0), Point3D(3, 5, 2)
        >>> p3, p4 = Point3D(-2, -2, -2), Point3D(0, 2, 1)
        >>> l1, l2, l3 = Line3D(p1, p2), Line3D(p1, p3), Line3D(p1, p4)
        >>> Line3D.are_concurrent(l1, l2, l3)
        True
        >>> l4 = Line3D(p2, p3)
        >>> Line3D.are_concurrent(l2, l3, l4)
        False

        """
        common_points = Intersection(*lines)
        if common_points.is_FiniteSet and len(common_points) == 1:
            return True
        return False

    def contains(self, other):
        """Subclasses should implement this method and should return
            True if other is on the boundaries of self;
            False if not on the boundaries of self;
            None if a determination cannot be made."""
        raise NotImplementedError()

    @property
    def direction(self):
        """The direction vector of the LinearEntity.

        Returns
        =======

        p : a Point; the ray from the origin to this point is the
            direction of `self`

        Examples
        ========

        >>> from sympy import Line
        >>> a, b = (1, 1), (1, 3)
        >>> Line(a, b).direction
        Point2D(0, 2)
        >>> Line(b, a).direction
        Point2D(0, -2)

        This can be reported so the distance from the origin is 1:

        >>> Line(b, a).direction.unit
        Point2D(0, -1)

        See Also
        ========

        sympy.geometry.point.Point.unit

        """
        return self.p2 - self.p1

    def intersection(self, other):
        """The intersection with another geometrical entity.

        Parameters
        ==========

        o : Point or LinearEntity

        Returns
        =======

        intersection : list of geometrical entities

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Line, Segment
        >>> p1, p2, p3 = Point(0, 0), Point(1, 1), Point(7, 7)
        >>> l1 = Line(p1, p2)
        >>> l1.intersection(p3)
        [Point2D(7, 7)]
        >>> p4, p5 = Point(5, 0), Point(0, 3)
        >>> l2 = Line(p4, p5)
        >>> l1.intersection(l2)
        [Point2D(15/8, 15/8)]
        >>> p6, p7 = Point(0, 5), Point(2, 6)
        >>> s1 = Segment(p6, p7)
        >>> l1.intersection(s1)
        []
        >>> from sympy import Point3D, Line3D, Segment3D
        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(1, 1, 1), Point3D(7, 7, 7)
        >>> l1 = Line3D(p1, p2)
        >>> l1.intersection(p3)
        [Point3D(7, 7, 7)]
        >>> l1 = Line3D(Point3D(4,19,12), Point3D(5,25,17))
        >>> l2 = Line3D(Point3D(-3, -15, -19), direction_ratio=[2,8,8])
        >>> l1.intersection(l2)
        [Point3D(1, 1, -3)]
        >>> p6, p7 = Point3D(0, 5, 2), Point3D(2, 6, 3)
        >>> s1 = Segment3D(p6, p7)
        >>> l1.intersection(s1)
        []

        """

        def intersect_parallel_rays(ray1, ray2):
            if ray1.direction.dot(ray2.direction) > 0:
                return [ray2] if ray1._span_test(ray2.p1) >= 0 else [ray1]
            else:
                st = ray1._span_test(ray2.p1)
                if st < 0:
                    return []
                elif st == 0:
                    return [ray2.p1]
                return [Segment(ray1.p1, ray2.p1)]

        def intersect_parallel_ray_and_segment(ray, seg):
            st1, st2 = (ray._span_test(seg.p1), ray._span_test(seg.p2))
            if st1 < 0 and st2 < 0:
                return []
            elif st1 >= 0 and st2 >= 0:
                return [seg]
            elif st1 >= 0:
                return [Segment(ray.p1, seg.p1)]
            else:
                return [Segment(ray.p1, seg.p2)]

        def intersect_parallel_segments(seg1, seg2):
            if seg1.contains(seg2):
                return [seg2]
            if seg2.contains(seg1):
                return [seg1]
            if seg1.direction.dot(seg2.direction) < 0:
                seg2 = Segment(seg2.p2, seg2.p1)
            if seg1._span_test(seg2.p1) < 0:
                seg1, seg2 = (seg2, seg1)
            if seg2._span_test(seg1.p2) < 0:
                return []
            return [Segment(seg2.p1, seg1.p2)]
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=self.ambient_dimension)
        if other.is_Point:
            if self.contains(other):
                return [other]
            else:
                return []
        elif isinstance(other, LinearEntity):
            pts = Point._normalize_dimension(self.p1, self.p2, other.p1, other.p2)
            rank = Point.affine_rank(*pts)
            if rank == 1:
                if isinstance(self, Line):
                    return [other]
                if isinstance(other, Line):
                    return [self]
                if isinstance(self, Ray) and isinstance(other, Ray):
                    return intersect_parallel_rays(self, other)
                if isinstance(self, Ray) and isinstance(other, Segment):
                    return intersect_parallel_ray_and_segment(self, other)
                if isinstance(self, Segment) and isinstance(other, Ray):
                    return intersect_parallel_ray_and_segment(other, self)
                if isinstance(self, Segment) and isinstance(other, Segment):
                    return intersect_parallel_segments(self, other)
            elif rank == 2:
                l1 = Line(*pts[:2])
                l2 = Line(*pts[2:])
                if l1.direction.is_scalar_multiple(l2.direction):
                    return []
                m = Matrix([l1.direction, -l2.direction]).transpose()
                v = Matrix([l2.p1 - l1.p1]).transpose()
                m_rref, pivots = m.col_insert(2, v).rref(simplify=True)
                if len(pivots) != 2:
                    raise GeometryError('Failed when solving Mx=b when M={} and b={}'.format(m, v))
                coeff = m_rref[0, 2]
                line_intersection = l1.direction * coeff + self.p1
                if isinstance(self, Line) and isinstance(other, Line):
                    return [line_intersection]
                if (isinstance(self, Line) or self.contains(line_intersection)) and other.contains(line_intersection):
                    return [line_intersection]
                if not self.atoms(Float) and (not other.atoms(Float)):
                    return []
                tu = solve(self.arbitrary_point(t) - other.arbitrary_point(u), t, u, dict=True)[0]

                def ok(p, l):
                    if isinstance(l, Line):
                        return True
                    if isinstance(l, Ray):
                        return p.is_nonnegative
                    if isinstance(l, Segment):
                        return p.is_nonnegative and (1 - p).is_nonnegative
                    raise ValueError('unexpected line type')
                if ok(tu[t], self) and ok(tu[u], other):
                    return [line_intersection]
                return []
            else:
                return []
        return other.intersection(self)

    def is_parallel(l1, l2):
        """Are two linear entities parallel?

        Parameters
        ==========

        l1 : LinearEntity
        l2 : LinearEntity

        Returns
        =======

        True : if l1 and l2 are parallel,
        False : otherwise.

        See Also
        ========

        coefficients

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(0, 0), Point(1, 1)
        >>> p3, p4 = Point(3, 4), Point(6, 7)
        >>> l1, l2 = Line(p1, p2), Line(p3, p4)
        >>> Line.is_parallel(l1, l2)
        True
        >>> p5 = Point(6, 6)
        >>> l3 = Line(p3, p5)
        >>> Line.is_parallel(l1, l3)
        False
        >>> from sympy import Point3D, Line3D
        >>> p1, p2 = Point3D(0, 0, 0), Point3D(3, 4, 5)
        >>> p3, p4 = Point3D(2, 1, 1), Point3D(8, 9, 11)
        >>> l1, l2 = Line3D(p1, p2), Line3D(p3, p4)
        >>> Line3D.is_parallel(l1, l2)
        True
        >>> p5 = Point3D(6, 6, 6)
        >>> l3 = Line3D(p3, p5)
        >>> Line3D.is_parallel(l1, l3)
        False

        """
        if not isinstance(l1, LinearEntity) and (not isinstance(l2, LinearEntity)):
            raise TypeError('Must pass only LinearEntity objects')
        return l1.direction.is_scalar_multiple(l2.direction)

    def is_perpendicular(l1, l2):
        """Are two linear entities perpendicular?

        Parameters
        ==========

        l1 : LinearEntity
        l2 : LinearEntity

        Returns
        =======

        True : if l1 and l2 are perpendicular,
        False : otherwise.

        See Also
        ========

        coefficients

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2, p3 = Point(0, 0), Point(1, 1), Point(-1, 1)
        >>> l1, l2 = Line(p1, p2), Line(p1, p3)
        >>> l1.is_perpendicular(l2)
        True
        >>> p4 = Point(5, 3)
        >>> l3 = Line(p1, p4)
        >>> l1.is_perpendicular(l3)
        False
        >>> from sympy import Point3D, Line3D
        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(1, 1, 1), Point3D(-1, 2, 0)
        >>> l1, l2 = Line3D(p1, p2), Line3D(p2, p3)
        >>> l1.is_perpendicular(l2)
        False
        >>> p4 = Point3D(5, 3, 7)
        >>> l3 = Line3D(p1, p4)
        >>> l1.is_perpendicular(l3)
        False

        """
        if not isinstance(l1, LinearEntity) and (not isinstance(l2, LinearEntity)):
            raise TypeError('Must pass only LinearEntity objects')
        return S.Zero.equals(l1.direction.dot(l2.direction))

    def is_similar(self, other):
        """
        Return True if self and other are contained in the same line.

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2, p3 = Point(0, 1), Point(3, 4), Point(2, 3)
        >>> l1 = Line(p1, p2)
        >>> l2 = Line(p1, p3)
        >>> l1.is_similar(l2)
        True
        """
        l = Line(self.p1, self.p2)
        return l.contains(other)

    @property
    def length(self):
        """
        The length of the line.

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(0, 0), Point(3, 5)
        >>> l1 = Line(p1, p2)
        >>> l1.length
        oo
        """
        return S.Infinity

    @property
    def p1(self):
        """The first defining point of a linear entity.

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(0, 0), Point(5, 3)
        >>> l = Line(p1, p2)
        >>> l.p1
        Point2D(0, 0)

        """
        return self.args[0]

    @property
    def p2(self):
        """The second defining point of a linear entity.

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(0, 0), Point(5, 3)
        >>> l = Line(p1, p2)
        >>> l.p2
        Point2D(5, 3)

        """
        return self.args[1]

    def parallel_line(self, p):
        """Create a new Line parallel to this linear entity which passes
        through the point `p`.

        Parameters
        ==========

        p : Point

        Returns
        =======

        line : Line

        See Also
        ========

        is_parallel

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2, p3 = Point(0, 0), Point(2, 3), Point(-2, 2)
        >>> l1 = Line(p1, p2)
        >>> l2 = l1.parallel_line(p3)
        >>> p3 in l2
        True
        >>> l1.is_parallel(l2)
        True
        >>> from sympy import Point3D, Line3D
        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(2, 3, 4), Point3D(-2, 2, 0)
        >>> l1 = Line3D(p1, p2)
        >>> l2 = l1.parallel_line(p3)
        >>> p3 in l2
        True
        >>> l1.is_parallel(l2)
        True

        """
        p = Point(p, dim=self.ambient_dimension)
        return Line(p, p + self.direction)

    def perpendicular_line(self, p):
        """Create a new Line perpendicular to this linear entity which passes
        through the point `p`.

        Parameters
        ==========

        p : Point

        Returns
        =======

        line : Line

        See Also
        ========

        sympy.geometry.line.LinearEntity.is_perpendicular, perpendicular_segment

        Examples
        ========

        >>> from sympy import Point3D, Line3D
        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(2, 3, 4), Point3D(-2, 2, 0)
        >>> L = Line3D(p1, p2)
        >>> P = L.perpendicular_line(p3); P
        Line3D(Point3D(-2, 2, 0), Point3D(4/29, 6/29, 8/29))
        >>> L.is_perpendicular(P)
        True

        In 3D the, the first point used to define the line is the point
        through which the perpendicular was required to pass; the
        second point is (arbitrarily) contained in the given line:

        >>> P.p2 in L
        True
        """
        p = Point(p, dim=self.ambient_dimension)
        if p in self:
            p = p + self.direction.orthogonal_direction
        return Line(p, self.projection(p))

    def perpendicular_segment(self, p):
        """Create a perpendicular line segment from `p` to this line.

        The endpoints of the segment are ``p`` and the closest point in
        the line containing self. (If self is not a line, the point might
        not be in self.)

        Parameters
        ==========

        p : Point

        Returns
        =======

        segment : Segment

        Notes
        =====

        Returns `p` itself if `p` is on this linear entity.

        See Also
        ========

        perpendicular_line

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2, p3 = Point(0, 0), Point(1, 1), Point(0, 2)
        >>> l1 = Line(p1, p2)
        >>> s1 = l1.perpendicular_segment(p3)
        >>> l1.is_perpendicular(s1)
        True
        >>> p3 in s1
        True
        >>> l1.perpendicular_segment(Point(4, 0))
        Segment2D(Point2D(4, 0), Point2D(2, 2))
        >>> from sympy import Point3D, Line3D
        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(1, 1, 1), Point3D(0, 2, 0)
        >>> l1 = Line3D(p1, p2)
        >>> s1 = l1.perpendicular_segment(p3)
        >>> l1.is_perpendicular(s1)
        True
        >>> p3 in s1
        True
        >>> l1.perpendicular_segment(Point3D(4, 0, 0))
        Segment3D(Point3D(4, 0, 0), Point3D(4/3, 4/3, 4/3))

        """
        p = Point(p, dim=self.ambient_dimension)
        if p in self:
            return p
        l = self.perpendicular_line(p)
        p2, = Intersection(Line(self.p1, self.p2), l)
        return Segment(p, p2)

    @property
    def points(self):
        """The two points used to define this linear entity.

        Returns
        =======

        points : tuple of Points

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(0, 0), Point(5, 11)
        >>> l1 = Line(p1, p2)
        >>> l1.points
        (Point2D(0, 0), Point2D(5, 11))

        """
        return (self.p1, self.p2)

    def projection(self, other):
        """Project a point, line, ray, or segment onto this linear entity.

        Parameters
        ==========

        other : Point or LinearEntity (Line, Ray, Segment)

        Returns
        =======

        projection : Point or LinearEntity (Line, Ray, Segment)
            The return type matches the type of the parameter ``other``.

        Raises
        ======

        GeometryError
            When method is unable to perform projection.

        Notes
        =====

        A projection involves taking the two points that define
        the linear entity and projecting those points onto a
        Line and then reforming the linear entity using these
        projections.
        A point P is projected onto a line L by finding the point
        on L that is closest to P. This point is the intersection
        of L and the line perpendicular to L that passes through P.

        See Also
        ========

        sympy.geometry.point.Point, perpendicular_line

        Examples
        ========

        >>> from sympy import Point, Line, Segment, Rational
        >>> p1, p2, p3 = Point(0, 0), Point(1, 1), Point(Rational(1, 2), 0)
        >>> l1 = Line(p1, p2)
        >>> l1.projection(p3)
        Point2D(1/4, 1/4)
        >>> p4, p5 = Point(10, 0), Point(12, 1)
        >>> s1 = Segment(p4, p5)
        >>> l1.projection(s1)
        Segment2D(Point2D(5, 5), Point2D(13/2, 13/2))
        >>> p1, p2, p3 = Point(0, 0, 1), Point(1, 1, 2), Point(2, 0, 1)
        >>> l1 = Line(p1, p2)
        >>> l1.projection(p3)
        Point3D(2/3, 2/3, 5/3)
        >>> p4, p5 = Point(10, 0, 1), Point(12, 1, 3)
        >>> s1 = Segment(p4, p5)
        >>> l1.projection(s1)
        Segment3D(Point3D(10/3, 10/3, 13/3), Point3D(5, 5, 6))

        """
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=self.ambient_dimension)

        def proj_point(p):
            return Point.project(p - self.p1, self.direction) + self.p1
        if isinstance(other, Point):
            return proj_point(other)
        elif isinstance(other, LinearEntity):
            p1, p2 = (proj_point(other.p1), proj_point(other.p2))
            if p1 == p2:
                return p1
            projected = other.__class__(p1, p2)
            projected = Intersection(self, projected)
            if projected.is_empty:
                return projected
            if projected.is_FiniteSet and len(projected) == 1:
                a, = projected
                return a
            if self.direction.dot(projected.direction) < 0:
                p1, p2 = projected.args
                projected = projected.func(p2, p1)
            return projected
        raise GeometryError('Do not know how to project %s onto %s' % (other, self))

    def random_point(self, seed=None):
        """A random point on a LinearEntity.

        Returns
        =======

        point : Point

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Line, Ray, Segment
        >>> p1, p2 = Point(0, 0), Point(5, 3)
        >>> line = Line(p1, p2)
        >>> r = line.random_point(seed=42)  # seed value is optional
        >>> r.n(3)
        Point2D(-0.72, -0.432)
        >>> r in line
        True
        >>> Ray(p1, p2).random_point(seed=42).n(3)
        Point2D(0.72, 0.432)
        >>> Segment(p1, p2).random_point(seed=42).n(3)
        Point2D(3.2, 1.92)

        """
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        pt = self.arbitrary_point(t)
        if isinstance(self, Ray):
            v = abs(rng.gauss(0, 1))
        elif isinstance(self, Segment):
            v = rng.random()
        elif isinstance(self, Line):
            v = rng.gauss(0, 1)
        else:
            raise NotImplementedError('unhandled line type')
        return pt.subs(t, Rational(v))

    def bisectors(self, other):
        """Returns the perpendicular lines which pass through the intersections
        of self and other that are in the same plane.

        Parameters
        ==========

        line : Line3D

        Returns
        =======

        list: two Line instances

        Examples
        ========

        >>> from sympy import Point3D, Line3D
        >>> r1 = Line3D(Point3D(0, 0, 0), Point3D(1, 0, 0))
        >>> r2 = Line3D(Point3D(0, 0, 0), Point3D(0, 1, 0))
        >>> r1.bisectors(r2)
        [Line3D(Point3D(0, 0, 0), Point3D(1, 1, 0)), Line3D(Point3D(0, 0, 0), Point3D(1, -1, 0))]

        """
        if not isinstance(other, LinearEntity):
            raise GeometryError('Expecting LinearEntity, not %s' % other)
        l1, l2 = (self, other)
        if l1.p1.ambient_dimension != l2.p1.ambient_dimension:
            if isinstance(l1, Line2D):
                l1, l2 = (l2, l1)
            _, p1 = Point._normalize_dimension(l1.p1, l2.p1, on_morph='ignore')
            _, p2 = Point._normalize_dimension(l1.p2, l2.p2, on_morph='ignore')
            l2 = Line(p1, p2)
        point = intersection(l1, l2)
        if not point:
            raise GeometryError('The lines do not intersect')
        else:
            pt = point[0]
            if isinstance(pt, Line):
                return [self]
        d1 = l1.direction.unit
        d2 = l2.direction.unit
        bis1 = Line(pt, pt + d1 + d2)
        bis2 = Line(pt, pt + d1 - d2)
        return [bis1, bis2]