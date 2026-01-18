import functools
import inspect
import math
from numbers import Number, Real
import textwrap
from types import SimpleNamespace
from collections import namedtuple
from matplotlib.transforms import Affine2D
import numpy as np
import matplotlib as mpl
from . import (_api, artist, cbook, colors, _docstring, hatch as mhatch,
from .bezier import (
from .path import Path
from ._enums import JoinStyle, CapStyle
@_docstring.dedent_interpd
class ConnectionStyle(_Style):
    """
    `ConnectionStyle` is a container class which defines
    several connectionstyle classes, which is used to create a path
    between two points.  These are mainly used with `FancyArrowPatch`.

    A connectionstyle object can be either created as::

           ConnectionStyle.Arc3(rad=0.2)

    or::

           ConnectionStyle("Arc3", rad=0.2)

    or::

           ConnectionStyle("Arc3, rad=0.2")

    The following classes are defined

    %(ConnectionStyle:table)s

    An instance of any connection style class is a callable object,
    whose call signature is::

        __call__(self, posA, posB,
                 patchA=None, patchB=None,
                 shrinkA=2., shrinkB=2.)

    and it returns a `.Path` instance. *posA* and *posB* are
    tuples of (x, y) coordinates of the two points to be
    connected. *patchA* (or *patchB*) is given, the returned path is
    clipped so that it start (or end) from the boundary of the
    patch. The path is further shrunk by *shrinkA* (or *shrinkB*)
    which is given in points.
    """
    _style_list = {}

    class _Base:
        """
        A base class for connectionstyle classes. The subclass needs
        to implement a *connect* method whose call signature is::

          connect(posA, posB)

        where posA and posB are tuples of x, y coordinates to be
        connected.  The method needs to return a path connecting two
        points. This base class defines a __call__ method, and a few
        helper methods.
        """

        @_api.deprecated('3.7')
        class SimpleEvent:

            def __init__(self, xy):
                self.x, self.y = xy

        def _in_patch(self, patch):
            """
            Return a predicate function testing whether a point *xy* is
            contained in *patch*.
            """
            return lambda xy: patch.contains(SimpleNamespace(x=xy[0], y=xy[1]))[0]

        def _clip(self, path, in_start, in_stop):
            """
            Clip *path* at its start by the region where *in_start* returns
            True, and at its stop by the region where *in_stop* returns True.

            The original path is assumed to start in the *in_start* region and
            to stop in the *in_stop* region.
            """
            if in_start:
                try:
                    _, path = split_path_inout(path, in_start)
                except ValueError:
                    pass
            if in_stop:
                try:
                    path, _ = split_path_inout(path, in_stop)
                except ValueError:
                    pass
            return path

        def __call__(self, posA, posB, shrinkA=2.0, shrinkB=2.0, patchA=None, patchB=None):
            """
            Call the *connect* method to create a path between *posA* and
            *posB*; then clip and shrink the path.
            """
            path = self.connect(posA, posB)
            path = self._clip(path, self._in_patch(patchA) if patchA else None, self._in_patch(patchB) if patchB else None)
            path = self._clip(path, inside_circle(*path.vertices[0], shrinkA) if shrinkA else None, inside_circle(*path.vertices[-1], shrinkB) if shrinkB else None)
            return path

    @_register_style(_style_list)
    class Arc3(_Base):
        """
        Creates a simple quadratic Bézier curve between two
        points. The curve is created so that the middle control point
        (C1) is located at the same distance from the start (C0) and
        end points(C2) and the distance of the C1 to the line
        connecting C0-C2 is *rad* times the distance of C0-C2.
        """

        def __init__(self, rad=0.0):
            """
            Parameters
            ----------
            rad : float
              Curvature of the curve.
            """
            self.rad = rad

        def connect(self, posA, posB):
            x1, y1 = posA
            x2, y2 = posB
            x12, y12 = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            dx, dy = (x2 - x1, y2 - y1)
            f = self.rad
            cx, cy = (x12 + f * dy, y12 - f * dx)
            vertices = [(x1, y1), (cx, cy), (x2, y2)]
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            return Path(vertices, codes)

    @_register_style(_style_list)
    class Angle3(_Base):
        """
        Creates a simple quadratic Bézier curve between two points. The middle
        control point is placed at the intersecting point of two lines which
        cross the start and end point, and have a slope of *angleA* and
        *angleB*, respectively.
        """

        def __init__(self, angleA=90, angleB=0):
            """
            Parameters
            ----------
            angleA : float
              Starting angle of the path.

            angleB : float
              Ending angle of the path.
            """
            self.angleA = angleA
            self.angleB = angleB

        def connect(self, posA, posB):
            x1, y1 = posA
            x2, y2 = posB
            cosA = math.cos(math.radians(self.angleA))
            sinA = math.sin(math.radians(self.angleA))
            cosB = math.cos(math.radians(self.angleB))
            sinB = math.sin(math.radians(self.angleB))
            cx, cy = get_intersection(x1, y1, cosA, sinA, x2, y2, cosB, sinB)
            vertices = [(x1, y1), (cx, cy), (x2, y2)]
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            return Path(vertices, codes)

    @_register_style(_style_list)
    class Angle(_Base):
        """
        Creates a piecewise continuous quadratic Bézier path between two
        points. The path has a one passing-through point placed at the
        intersecting point of two lines which cross the start and end point,
        and have a slope of *angleA* and *angleB*, respectively.
        The connecting edges are rounded with *rad*.
        """

        def __init__(self, angleA=90, angleB=0, rad=0.0):
            """
            Parameters
            ----------
            angleA : float
              Starting angle of the path.

            angleB : float
              Ending angle of the path.

            rad : float
              Rounding radius of the edge.
            """
            self.angleA = angleA
            self.angleB = angleB
            self.rad = rad

        def connect(self, posA, posB):
            x1, y1 = posA
            x2, y2 = posB
            cosA = math.cos(math.radians(self.angleA))
            sinA = math.sin(math.radians(self.angleA))
            cosB = math.cos(math.radians(self.angleB))
            sinB = math.sin(math.radians(self.angleB))
            cx, cy = get_intersection(x1, y1, cosA, sinA, x2, y2, cosB, sinB)
            vertices = [(x1, y1)]
            codes = [Path.MOVETO]
            if self.rad == 0.0:
                vertices.append((cx, cy))
                codes.append(Path.LINETO)
            else:
                dx1, dy1 = (x1 - cx, y1 - cy)
                d1 = np.hypot(dx1, dy1)
                f1 = self.rad / d1
                dx2, dy2 = (x2 - cx, y2 - cy)
                d2 = np.hypot(dx2, dy2)
                f2 = self.rad / d2
                vertices.extend([(cx + dx1 * f1, cy + dy1 * f1), (cx, cy), (cx + dx2 * f2, cy + dy2 * f2)])
                codes.extend([Path.LINETO, Path.CURVE3, Path.CURVE3])
            vertices.append((x2, y2))
            codes.append(Path.LINETO)
            return Path(vertices, codes)

    @_register_style(_style_list)
    class Arc(_Base):
        """
        Creates a piecewise continuous quadratic Bézier path between two
        points. The path can have two passing-through points, a
        point placed at the distance of *armA* and angle of *angleA* from
        point A, another point with respect to point B. The edges are
        rounded with *rad*.
        """

        def __init__(self, angleA=0, angleB=0, armA=None, armB=None, rad=0.0):
            """
            Parameters
            ----------
            angleA : float
              Starting angle of the path.

            angleB : float
              Ending angle of the path.

            armA : float or None
              Length of the starting arm.

            armB : float or None
              Length of the ending arm.

            rad : float
              Rounding radius of the edges.
            """
            self.angleA = angleA
            self.angleB = angleB
            self.armA = armA
            self.armB = armB
            self.rad = rad

        def connect(self, posA, posB):
            x1, y1 = posA
            x2, y2 = posB
            vertices = [(x1, y1)]
            rounded = []
            codes = [Path.MOVETO]
            if self.armA:
                cosA = math.cos(math.radians(self.angleA))
                sinA = math.sin(math.radians(self.angleA))
                d = self.armA - self.rad
                rounded.append((x1 + d * cosA, y1 + d * sinA))
                d = self.armA
                rounded.append((x1 + d * cosA, y1 + d * sinA))
            if self.armB:
                cosB = math.cos(math.radians(self.angleB))
                sinB = math.sin(math.radians(self.angleB))
                x_armB, y_armB = (x2 + self.armB * cosB, y2 + self.armB * sinB)
                if rounded:
                    xp, yp = rounded[-1]
                    dx, dy = (x_armB - xp, y_armB - yp)
                    dd = (dx * dx + dy * dy) ** 0.5
                    rounded.append((xp + self.rad * dx / dd, yp + self.rad * dy / dd))
                    vertices.extend(rounded)
                    codes.extend([Path.LINETO, Path.CURVE3, Path.CURVE3])
                else:
                    xp, yp = vertices[-1]
                    dx, dy = (x_armB - xp, y_armB - yp)
                    dd = (dx * dx + dy * dy) ** 0.5
                d = dd - self.rad
                rounded = [(xp + d * dx / dd, yp + d * dy / dd), (x_armB, y_armB)]
            if rounded:
                xp, yp = rounded[-1]
                dx, dy = (x2 - xp, y2 - yp)
                dd = (dx * dx + dy * dy) ** 0.5
                rounded.append((xp + self.rad * dx / dd, yp + self.rad * dy / dd))
                vertices.extend(rounded)
                codes.extend([Path.LINETO, Path.CURVE3, Path.CURVE3])
            vertices.append((x2, y2))
            codes.append(Path.LINETO)
            return Path(vertices, codes)

    @_register_style(_style_list)
    class Bar(_Base):
        """
        A line with *angle* between A and B with *armA* and *armB*. One of the
        arms is extended so that they are connected in a right angle. The
        length of *armA* is determined by (*armA* + *fraction* x AB distance).
        Same for *armB*.
        """

        def __init__(self, armA=0.0, armB=0.0, fraction=0.3, angle=None):
            """
            Parameters
            ----------
            armA : float
                Minimum length of armA.

            armB : float
                Minimum length of armB.

            fraction : float
                A fraction of the distance between two points that will be
                added to armA and armB.

            angle : float or None
                Angle of the connecting line (if None, parallel to A and B).
            """
            self.armA = armA
            self.armB = armB
            self.fraction = fraction
            self.angle = angle

        def connect(self, posA, posB):
            x1, y1 = posA
            x20, y20 = x2, y2 = posB
            theta1 = math.atan2(y2 - y1, x2 - x1)
            dx, dy = (x2 - x1, y2 - y1)
            dd = (dx * dx + dy * dy) ** 0.5
            ddx, ddy = (dx / dd, dy / dd)
            armA, armB = (self.armA, self.armB)
            if self.angle is not None:
                theta0 = np.deg2rad(self.angle)
                dtheta = theta1 - theta0
                dl = dd * math.sin(dtheta)
                dL = dd * math.cos(dtheta)
                x2, y2 = (x1 + dL * math.cos(theta0), y1 + dL * math.sin(theta0))
                armB = armB - dl
                dx, dy = (x2 - x1, y2 - y1)
                dd2 = (dx * dx + dy * dy) ** 0.5
                ddx, ddy = (dx / dd2, dy / dd2)
            arm = max(armA, armB)
            f = self.fraction * dd + arm
            cx1, cy1 = (x1 + f * ddy, y1 - f * ddx)
            cx2, cy2 = (x2 + f * ddy, y2 - f * ddx)
            vertices = [(x1, y1), (cx1, cy1), (cx2, cy2), (x20, y20)]
            codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]
            return Path(vertices, codes)