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
class RegularPolygon(Polygon):
    """
    A regular polygon.

    Such a polygon has all internal angles equal and all sides the same length.

    Parameters
    ==========

    center : Point
    radius : number or Basic instance
        The distance from the center to a vertex
    n : int
        The number of sides

    Attributes
    ==========

    vertices
    center
    radius
    rotation
    apothem
    interior_angle
    exterior_angle
    circumcircle
    incircle
    angles

    Raises
    ======

    GeometryError
        If the `center` is not a Point, or the `radius` is not a number or Basic
        instance, or the number of sides, `n`, is less than three.

    Notes
    =====

    A RegularPolygon can be instantiated with Polygon with the kwarg n.

    Regular polygons are instantiated with a center, radius, number of sides
    and a rotation angle. Whereas the arguments of a Polygon are vertices, the
    vertices of the RegularPolygon must be obtained with the vertices method.

    See Also
    ========

    sympy.geometry.point.Point, Polygon

    Examples
    ========

    >>> from sympy import RegularPolygon, Point
    >>> r = RegularPolygon(Point(0, 0), 5, 3)
    >>> r
    RegularPolygon(Point2D(0, 0), 5, 3, 0)
    >>> r.vertices[0]
    Point2D(5, 0)

    """
    __slots__ = ('_n', '_center', '_radius', '_rot')

    def __new__(self, c, r, n, rot=0, **kwargs):
        r, n, rot = map(sympify, (r, n, rot))
        c = Point(c, dim=2, **kwargs)
        if not isinstance(r, Expr):
            raise GeometryError('r must be an Expr object, not %s' % r)
        if n.is_Number:
            as_int(n)
            if n < 3:
                raise GeometryError('n must be a >= 3, not %s' % n)
        obj = GeometryEntity.__new__(self, c, r, n, **kwargs)
        obj._n = n
        obj._center = c
        obj._radius = r
        obj._rot = rot % (2 * S.Pi / n) if rot.is_number else rot
        return obj

    def _eval_evalf(self, prec=15, **options):
        c, r, n, a = self.args
        dps = prec_to_dps(prec)
        c, r, a = [i.evalf(n=dps, **options) for i in (c, r, a)]
        return self.func(c, r, n, a)

    @property
    def args(self):
        """
        Returns the center point, the radius,
        the number of sides, and the orientation angle.

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> r = RegularPolygon(Point(0, 0), 5, 3)
        >>> r.args
        (Point2D(0, 0), 5, 3, 0)
        """
        return (self._center, self._radius, self._n, self._rot)

    def __str__(self):
        return 'RegularPolygon(%s, %s, %s, %s)' % tuple(self.args)

    def __repr__(self):
        return 'RegularPolygon(%s, %s, %s, %s)' % tuple(self.args)

    @property
    def area(self):
        """Returns the area.

        Examples
        ========

        >>> from sympy import RegularPolygon
        >>> square = RegularPolygon((0, 0), 1, 4)
        >>> square.area
        2
        >>> _ == square.length**2
        True
        """
        c, r, n, rot = self.args
        return sign(r) * n * self.length ** 2 / (4 * tan(pi / n))

    @property
    def length(self):
        """Returns the length of the sides.

        The half-length of the side and the apothem form two legs
        of a right triangle whose hypotenuse is the radius of the
        regular polygon.

        Examples
        ========

        >>> from sympy import RegularPolygon
        >>> from sympy import sqrt
        >>> s = square_in_unit_circle = RegularPolygon((0, 0), 1, 4)
        >>> s.length
        sqrt(2)
        >>> sqrt((_/2)**2 + s.apothem**2) == s.radius
        True

        """
        return self.radius * 2 * sin(pi / self._n)

    @property
    def center(self):
        """The center of the RegularPolygon

        This is also the center of the circumscribing circle.

        Returns
        =======

        center : Point

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.ellipse.Ellipse.center

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> rp = RegularPolygon(Point(0, 0), 5, 4)
        >>> rp.center
        Point2D(0, 0)
        """
        return self._center
    centroid = center

    @property
    def circumcenter(self):
        """
        Alias for center.

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> rp = RegularPolygon(Point(0, 0), 5, 4)
        >>> rp.circumcenter
        Point2D(0, 0)
        """
        return self.center

    @property
    def radius(self):
        """Radius of the RegularPolygon

        This is also the radius of the circumscribing circle.

        Returns
        =======

        radius : number or instance of Basic

        See Also
        ========

        sympy.geometry.line.Segment.length, sympy.geometry.ellipse.Circle.radius

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy import RegularPolygon, Point
        >>> radius = Symbol('r')
        >>> rp = RegularPolygon(Point(0, 0), radius, 4)
        >>> rp.radius
        r

        """
        return self._radius

    @property
    def circumradius(self):
        """
        Alias for radius.

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy import RegularPolygon, Point
        >>> radius = Symbol('r')
        >>> rp = RegularPolygon(Point(0, 0), radius, 4)
        >>> rp.circumradius
        r
        """
        return self.radius

    @property
    def rotation(self):
        """CCW angle by which the RegularPolygon is rotated

        Returns
        =======

        rotation : number or instance of Basic

        Examples
        ========

        >>> from sympy import pi
        >>> from sympy.abc import a
        >>> from sympy import RegularPolygon, Point
        >>> RegularPolygon(Point(0, 0), 3, 4, pi/4).rotation
        pi/4

        Numerical rotation angles are made canonical:

        >>> RegularPolygon(Point(0, 0), 3, 4, a).rotation
        a
        >>> RegularPolygon(Point(0, 0), 3, 4, pi).rotation
        0

        """
        return self._rot

    @property
    def apothem(self):
        """The inradius of the RegularPolygon.

        The apothem/inradius is the radius of the inscribed circle.

        Returns
        =======

        apothem : number or instance of Basic

        See Also
        ========

        sympy.geometry.line.Segment.length, sympy.geometry.ellipse.Circle.radius

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy import RegularPolygon, Point
        >>> radius = Symbol('r')
        >>> rp = RegularPolygon(Point(0, 0), radius, 4)
        >>> rp.apothem
        sqrt(2)*r/2

        """
        return self.radius * cos(S.Pi / self._n)

    @property
    def inradius(self):
        """
        Alias for apothem.

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy import RegularPolygon, Point
        >>> radius = Symbol('r')
        >>> rp = RegularPolygon(Point(0, 0), radius, 4)
        >>> rp.inradius
        sqrt(2)*r/2
        """
        return self.apothem

    @property
    def interior_angle(self):
        """Measure of the interior angles.

        Returns
        =======

        interior_angle : number

        See Also
        ========

        sympy.geometry.line.LinearEntity.angle_between

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> rp = RegularPolygon(Point(0, 0), 4, 8)
        >>> rp.interior_angle
        3*pi/4

        """
        return (self._n - 2) * S.Pi / self._n

    @property
    def exterior_angle(self):
        """Measure of the exterior angles.

        Returns
        =======

        exterior_angle : number

        See Also
        ========

        sympy.geometry.line.LinearEntity.angle_between

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> rp = RegularPolygon(Point(0, 0), 4, 8)
        >>> rp.exterior_angle
        pi/4

        """
        return 2 * S.Pi / self._n

    @property
    def circumcircle(self):
        """The circumcircle of the RegularPolygon.

        Returns
        =======

        circumcircle : Circle

        See Also
        ========

        circumcenter, sympy.geometry.ellipse.Circle

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> rp = RegularPolygon(Point(0, 0), 4, 8)
        >>> rp.circumcircle
        Circle(Point2D(0, 0), 4)

        """
        return Circle(self.center, self.radius)

    @property
    def incircle(self):
        """The incircle of the RegularPolygon.

        Returns
        =======

        incircle : Circle

        See Also
        ========

        inradius, sympy.geometry.ellipse.Circle

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> rp = RegularPolygon(Point(0, 0), 4, 7)
        >>> rp.incircle
        Circle(Point2D(0, 0), 4*cos(pi/7))

        """
        return Circle(self.center, self.apothem)

    @property
    def angles(self):
        """
        Returns a dictionary with keys, the vertices of the Polygon,
        and values, the interior angle at each vertex.

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> r = RegularPolygon(Point(0, 0), 5, 3)
        >>> r.angles
        {Point2D(-5/2, -5*sqrt(3)/2): pi/3,
         Point2D(-5/2, 5*sqrt(3)/2): pi/3,
         Point2D(5, 0): pi/3}
        """
        ret = {}
        ang = self.interior_angle
        for v in self.vertices:
            ret[v] = ang
        return ret

    def encloses_point(self, p):
        """
        Return True if p is enclosed by (is inside of) self.

        Notes
        =====

        Being on the border of self is considered False.

        The general Polygon.encloses_point method is called only if
        a point is not within or beyond the incircle or circumcircle,
        respectively.

        Parameters
        ==========

        p : Point

        Returns
        =======

        encloses_point : True, False or None

        See Also
        ========

        sympy.geometry.ellipse.Ellipse.encloses_point

        Examples
        ========

        >>> from sympy import RegularPolygon, S, Point, Symbol
        >>> p = RegularPolygon((0, 0), 3, 4)
        >>> p.encloses_point(Point(0, 0))
        True
        >>> r, R = p.inradius, p.circumradius
        >>> p.encloses_point(Point((r + R)/2, 0))
        True
        >>> p.encloses_point(Point(R/2, R/2 + (R - r)/10))
        False
        >>> t = Symbol('t', real=True)
        >>> p.encloses_point(p.arbitrary_point().subs(t, S.Half))
        False
        >>> p.encloses_point(Point(5, 5))
        False

        """
        c = self.center
        d = Segment(c, p).length
        if d >= self.radius:
            return False
        elif d < self.inradius:
            return True
        else:
            return Polygon.encloses_point(self, p)

    def spin(self, angle):
        """Increment *in place* the virtual Polygon's rotation by ccw angle.

        See also: rotate method which moves the center.

        >>> from sympy import Polygon, Point, pi
        >>> r = Polygon(Point(0,0), 1, n=3)
        >>> r.vertices[0]
        Point2D(1, 0)
        >>> r.spin(pi/6)
        >>> r.vertices[0]
        Point2D(sqrt(3)/2, 1/2)

        See Also
        ========

        rotation
        rotate : Creates a copy of the RegularPolygon rotated about a Point

        """
        self._rot += angle

    def rotate(self, angle, pt=None):
        """Override GeometryEntity.rotate to first rotate the RegularPolygon
        about its center.

        >>> from sympy import Point, RegularPolygon, pi
        >>> t = RegularPolygon(Point(1, 0), 1, 3)
        >>> t.vertices[0] # vertex on x-axis
        Point2D(2, 0)
        >>> t.rotate(pi/2).vertices[0] # vertex on y axis now
        Point2D(0, 2)

        See Also
        ========

        rotation
        spin : Rotates a RegularPolygon in place

        """
        r = type(self)(*self.args)
        r._rot += angle
        return GeometryEntity.rotate(r, angle, pt)

    def scale(self, x=1, y=1, pt=None):
        """Override GeometryEntity.scale since it is the radius that must be
        scaled (if x == y) or else a new Polygon must be returned.

        >>> from sympy import RegularPolygon

        Symmetric scaling returns a RegularPolygon:

        >>> RegularPolygon((0, 0), 1, 4).scale(2, 2)
        RegularPolygon(Point2D(0, 0), 2, 4, 0)

        Asymmetric scaling returns a kite as a Polygon:

        >>> RegularPolygon((0, 0), 1, 4).scale(2, 1)
        Polygon(Point2D(2, 0), Point2D(0, 1), Point2D(-2, 0), Point2D(0, -1))

        """
        if pt:
            pt = Point(pt, dim=2)
            return self.translate(*(-pt).args).scale(x, y).translate(*pt.args)
        if x != y:
            return Polygon(*self.vertices).scale(x, y)
        c, r, n, rot = self.args
        r *= x
        return self.func(c, r, n, rot)

    def reflect(self, line):
        """Override GeometryEntity.reflect since this is not made of only
        points.

        Examples
        ========

        >>> from sympy import RegularPolygon, Line

        >>> RegularPolygon((0, 0), 1, 4).reflect(Line((0, 1), slope=-2))
        RegularPolygon(Point2D(4/5, 2/5), -1, 4, atan(4/3))

        """
        c, r, n, rot = self.args
        v = self.vertices[0]
        d = v - c
        cc = c.reflect(line)
        vv = v.reflect(line)
        dd = vv - cc
        l1 = Ray((0, 0), dd)
        l2 = Ray((0, 0), d)
        ang = l1.closing_angle(l2)
        rot += ang
        return self.func(cc, -r, n, rot)

    @property
    def vertices(self):
        """The vertices of the RegularPolygon.

        Returns
        =======

        vertices : list
            Each vertex is a Point.

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> rp = RegularPolygon(Point(0, 0), 5, 4)
        >>> rp.vertices
        [Point2D(5, 0), Point2D(0, 5), Point2D(-5, 0), Point2D(0, -5)]

        """
        c = self._center
        r = abs(self._radius)
        rot = self._rot
        v = 2 * S.Pi / self._n
        return [Point(c.x + r * cos(k * v + rot), c.y + r * sin(k * v + rot)) for k in range(self._n)]

    def __eq__(self, o):
        if not isinstance(o, Polygon):
            return False
        elif not isinstance(o, RegularPolygon):
            return Polygon.__eq__(o, self)
        return self.args == o.args

    def __hash__(self):
        return super().__hash__()