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
class ArrowStyle(_Style):
    """
    `ArrowStyle` is a container class which defines several
    arrowstyle classes, which is used to create an arrow path along a
    given path.  These are mainly used with `FancyArrowPatch`.

    An arrowstyle object can be either created as::

           ArrowStyle.Fancy(head_length=.4, head_width=.4, tail_width=.4)

    or::

           ArrowStyle("Fancy", head_length=.4, head_width=.4, tail_width=.4)

    or::

           ArrowStyle("Fancy, head_length=.4, head_width=.4, tail_width=.4")

    The following classes are defined

    %(ArrowStyle:table)s

    For an overview of the visual appearance, see
    :doc:`/gallery/text_labels_and_annotations/fancyarrow_demo`.

    An instance of any arrow style class is a callable object,
    whose call signature is::

        __call__(self, path, mutation_size, linewidth, aspect_ratio=1.)

    and it returns a tuple of a `.Path` instance and a boolean
    value. *path* is a `.Path` instance along which the arrow
    will be drawn. *mutation_size* and *aspect_ratio* have the same
    meaning as in `BoxStyle`. *linewidth* is a line width to be
    stroked. This is meant to be used to correct the location of the
    head so that it does not overshoot the destination point, but not all
    classes support it.

    Notes
    -----
    *angleA* and *angleB* specify the orientation of the bracket, as either a
    clockwise or counterclockwise angle depending on the arrow type. 0 degrees
    means perpendicular to the line connecting the arrow's head and tail.

    .. plot:: gallery/text_labels_and_annotations/angles_on_bracket_arrows.py
    """
    _style_list = {}

    class _Base:
        """
        Arrow Transmuter Base class

        ArrowTransmuterBase and its derivatives are used to make a fancy
        arrow around a given path. The __call__ method returns a path
        (which will be used to create a PathPatch instance) and a boolean
        value indicating the path is open therefore is not fillable.  This
        class is not an artist and actual drawing of the fancy arrow is
        done by the FancyArrowPatch class.
        """

        @staticmethod
        def ensure_quadratic_bezier(path):
            """
            Some ArrowStyle classes only works with a simple quadratic
            Bézier curve (created with `.ConnectionStyle.Arc3` or
            `.ConnectionStyle.Angle3`). This static method checks if the
            provided path is a simple quadratic Bézier curve and returns its
            control points if true.
            """
            segments = list(path.iter_segments())
            if len(segments) != 2 or segments[0][1] != Path.MOVETO or segments[1][1] != Path.CURVE3:
                raise ValueError("'path' is not a valid quadratic Bezier curve")
            return [*segments[0][0], *segments[1][0]]

        def transmute(self, path, mutation_size, linewidth):
            """
            The transmute method is the very core of the ArrowStyle class and
            must be overridden in the subclasses. It receives the *path*
            object along which the arrow will be drawn, and the
            *mutation_size*, with which the arrow head etc. will be scaled.
            The *linewidth* may be used to adjust the path so that it does not
            pass beyond the given points. It returns a tuple of a `.Path`
            instance and a boolean. The boolean value indicate whether the
            path can be filled or not. The return value can also be a list of
            paths and list of booleans of the same length.
            """
            raise NotImplementedError('Derived must override')

        def __call__(self, path, mutation_size, linewidth, aspect_ratio=1.0):
            """
            The __call__ method is a thin wrapper around the transmute method
            and takes care of the aspect ratio.
            """
            if aspect_ratio is not None:
                vertices = path.vertices / [1, aspect_ratio]
                path_shrunk = Path(vertices, path.codes)
                path_mutated, fillable = self.transmute(path_shrunk, mutation_size, linewidth)
                if np.iterable(fillable):
                    path_list = [Path(p.vertices * [1, aspect_ratio], p.codes) for p in path_mutated]
                    return (path_list, fillable)
                else:
                    return (path_mutated, fillable)
            else:
                return self.transmute(path, mutation_size, linewidth)

    class _Curve(_Base):
        """
        A simple arrow which will work with any path instance. The
        returned path is the concatenation of the original path, and at
        most two paths representing the arrow head or bracket at the start
        point and at the end point. The arrow heads can be either open
        or closed.
        """
        arrow = '-'
        fillbegin = fillend = False

        def __init__(self, head_length=0.4, head_width=0.2, widthA=1.0, widthB=1.0, lengthA=0.2, lengthB=0.2, angleA=0, angleB=0, scaleA=None, scaleB=None):
            """
            Parameters
            ----------
            head_length : float, default: 0.4
                Length of the arrow head, relative to *mutation_size*.
            head_width : float, default: 0.2
                Width of the arrow head, relative to *mutation_size*.
            widthA, widthB : float, default: 1.0
                Width of the bracket.
            lengthA, lengthB : float, default: 0.2
                Length of the bracket.
            angleA, angleB : float, default: 0
                Orientation of the bracket, as a counterclockwise angle.
                0 degrees means perpendicular to the line.
            scaleA, scaleB : float, default: *mutation_size*
                The scale of the brackets.
            """
            self.head_length, self.head_width = (head_length, head_width)
            self.widthA, self.widthB = (widthA, widthB)
            self.lengthA, self.lengthB = (lengthA, lengthB)
            self.angleA, self.angleB = (angleA, angleB)
            self.scaleA, self.scaleB = (scaleA, scaleB)
            self._beginarrow_head = False
            self._beginarrow_bracket = False
            self._endarrow_head = False
            self._endarrow_bracket = False
            if '-' not in self.arrow:
                raise ValueError("arrow must have the '-' between the two heads")
            beginarrow, endarrow = self.arrow.split('-', 1)
            if beginarrow == '<':
                self._beginarrow_head = True
                self._beginarrow_bracket = False
            elif beginarrow == '<|':
                self._beginarrow_head = True
                self._beginarrow_bracket = False
                self.fillbegin = True
            elif beginarrow in (']', '|'):
                self._beginarrow_head = False
                self._beginarrow_bracket = True
            if endarrow == '>':
                self._endarrow_head = True
                self._endarrow_bracket = False
            elif endarrow == '|>':
                self._endarrow_head = True
                self._endarrow_bracket = False
                self.fillend = True
            elif endarrow in ('[', '|'):
                self._endarrow_head = False
                self._endarrow_bracket = True
            super().__init__()

        def _get_arrow_wedge(self, x0, y0, x1, y1, head_dist, cos_t, sin_t, linewidth):
            """
            Return the paths for arrow heads. Since arrow lines are
            drawn with capstyle=projected, The arrow goes beyond the
            desired point. This method also returns the amount of the path
            to be shrunken so that it does not overshoot.
            """
            dx, dy = (x0 - x1, y0 - y1)
            cp_distance = np.hypot(dx, dy)
            pad_projected = 0.5 * linewidth / sin_t
            if cp_distance == 0:
                cp_distance = 1
            ddx = pad_projected * dx / cp_distance
            ddy = pad_projected * dy / cp_distance
            dx = dx / cp_distance * head_dist
            dy = dy / cp_distance * head_dist
            dx1, dy1 = (cos_t * dx + sin_t * dy, -sin_t * dx + cos_t * dy)
            dx2, dy2 = (cos_t * dx - sin_t * dy, sin_t * dx + cos_t * dy)
            vertices_arrow = [(x1 + ddx + dx1, y1 + ddy + dy1), (x1 + ddx, y1 + ddy), (x1 + ddx + dx2, y1 + ddy + dy2)]
            codes_arrow = [Path.MOVETO, Path.LINETO, Path.LINETO]
            return (vertices_arrow, codes_arrow, ddx, ddy)

        def _get_bracket(self, x0, y0, x1, y1, width, length, angle):
            cos_t, sin_t = get_cos_sin(x1, y1, x0, y0)
            from matplotlib.bezier import get_normal_points
            x1, y1, x2, y2 = get_normal_points(x0, y0, cos_t, sin_t, width)
            dx, dy = (length * cos_t, length * sin_t)
            vertices_arrow = [(x1 + dx, y1 + dy), (x1, y1), (x2, y2), (x2 + dx, y2 + dy)]
            codes_arrow = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]
            if angle:
                trans = transforms.Affine2D().rotate_deg_around(x0, y0, angle)
                vertices_arrow = trans.transform(vertices_arrow)
            return (vertices_arrow, codes_arrow)

        def transmute(self, path, mutation_size, linewidth):
            if self._beginarrow_head or self._endarrow_head:
                head_length = self.head_length * mutation_size
                head_width = self.head_width * mutation_size
                head_dist = np.hypot(head_length, head_width)
                cos_t, sin_t = (head_length / head_dist, head_width / head_dist)
            scaleA = mutation_size if self.scaleA is None else self.scaleA
            scaleB = mutation_size if self.scaleB is None else self.scaleB
            x0, y0 = path.vertices[0]
            x1, y1 = path.vertices[1]
            has_begin_arrow = self._beginarrow_head and (x0, y0) != (x1, y1)
            verticesA, codesA, ddxA, ddyA = self._get_arrow_wedge(x1, y1, x0, y0, head_dist, cos_t, sin_t, linewidth) if has_begin_arrow else ([], [], 0, 0)
            x2, y2 = path.vertices[-2]
            x3, y3 = path.vertices[-1]
            has_end_arrow = self._endarrow_head and (x2, y2) != (x3, y3)
            verticesB, codesB, ddxB, ddyB = self._get_arrow_wedge(x2, y2, x3, y3, head_dist, cos_t, sin_t, linewidth) if has_end_arrow else ([], [], 0, 0)
            paths = [Path(np.concatenate([[(x0 + ddxA, y0 + ddyA)], path.vertices[1:-1], [(x3 + ddxB, y3 + ddyB)]]), path.codes)]
            fills = [False]
            if has_begin_arrow:
                if self.fillbegin:
                    paths.append(Path([*verticesA, (0, 0)], [*codesA, Path.CLOSEPOLY]))
                    fills.append(True)
                else:
                    paths.append(Path(verticesA, codesA))
                    fills.append(False)
            elif self._beginarrow_bracket:
                x0, y0 = path.vertices[0]
                x1, y1 = path.vertices[1]
                verticesA, codesA = self._get_bracket(x0, y0, x1, y1, self.widthA * scaleA, self.lengthA * scaleA, self.angleA)
                paths.append(Path(verticesA, codesA))
                fills.append(False)
            if has_end_arrow:
                if self.fillend:
                    fills.append(True)
                    paths.append(Path([*verticesB, (0, 0)], [*codesB, Path.CLOSEPOLY]))
                else:
                    fills.append(False)
                    paths.append(Path(verticesB, codesB))
            elif self._endarrow_bracket:
                x0, y0 = path.vertices[-1]
                x1, y1 = path.vertices[-2]
                verticesB, codesB = self._get_bracket(x0, y0, x1, y1, self.widthB * scaleB, self.lengthB * scaleB, self.angleB)
                paths.append(Path(verticesB, codesB))
                fills.append(False)
            return (paths, fills)

    @_register_style(_style_list, name='-')
    class Curve(_Curve):
        """A simple curve without any arrow head."""

        def __init__(self):
            super().__init__(head_length=0.2, head_width=0.1)

    @_register_style(_style_list, name='<-')
    class CurveA(_Curve):
        """An arrow with a head at its start point."""
        arrow = '<-'

    @_register_style(_style_list, name='->')
    class CurveB(_Curve):
        """An arrow with a head at its end point."""
        arrow = '->'

    @_register_style(_style_list, name='<->')
    class CurveAB(_Curve):
        """An arrow with heads both at the start and the end point."""
        arrow = '<->'

    @_register_style(_style_list, name='<|-')
    class CurveFilledA(_Curve):
        """An arrow with filled triangle head at the start."""
        arrow = '<|-'

    @_register_style(_style_list, name='-|>')
    class CurveFilledB(_Curve):
        """An arrow with filled triangle head at the end."""
        arrow = '-|>'

    @_register_style(_style_list, name='<|-|>')
    class CurveFilledAB(_Curve):
        """An arrow with filled triangle heads at both ends."""
        arrow = '<|-|>'

    @_register_style(_style_list, name=']-')
    class BracketA(_Curve):
        """An arrow with an outward square bracket at its start."""
        arrow = ']-'

        def __init__(self, widthA=1.0, lengthA=0.2, angleA=0):
            """
            Parameters
            ----------
            widthA : float, default: 1.0
                Width of the bracket.
            lengthA : float, default: 0.2
                Length of the bracket.
            angleA : float, default: 0 degrees
                Orientation of the bracket, as a counterclockwise angle.
                0 degrees means perpendicular to the line.
            """
            super().__init__(widthA=widthA, lengthA=lengthA, angleA=angleA)

    @_register_style(_style_list, name='-[')
    class BracketB(_Curve):
        """An arrow with an outward square bracket at its end."""
        arrow = '-['

        def __init__(self, widthB=1.0, lengthB=0.2, angleB=0):
            """
            Parameters
            ----------
            widthB : float, default: 1.0
                Width of the bracket.
            lengthB : float, default: 0.2
                Length of the bracket.
            angleB : float, default: 0 degrees
                Orientation of the bracket, as a counterclockwise angle.
                0 degrees means perpendicular to the line.
            """
            super().__init__(widthB=widthB, lengthB=lengthB, angleB=angleB)

    @_register_style(_style_list, name=']-[')
    class BracketAB(_Curve):
        """An arrow with outward square brackets at both ends."""
        arrow = ']-['

        def __init__(self, widthA=1.0, lengthA=0.2, angleA=0, widthB=1.0, lengthB=0.2, angleB=0):
            """
            Parameters
            ----------
            widthA, widthB : float, default: 1.0
                Width of the bracket.
            lengthA, lengthB : float, default: 0.2
                Length of the bracket.
            angleA, angleB : float, default: 0 degrees
                Orientation of the bracket, as a counterclockwise angle.
                0 degrees means perpendicular to the line.
            """
            super().__init__(widthA=widthA, lengthA=lengthA, angleA=angleA, widthB=widthB, lengthB=lengthB, angleB=angleB)

    @_register_style(_style_list, name='|-|')
    class BarAB(_Curve):
        """An arrow with vertical bars ``|`` at both ends."""
        arrow = '|-|'

        def __init__(self, widthA=1.0, angleA=0, widthB=1.0, angleB=0):
            """
            Parameters
            ----------
            widthA, widthB : float, default: 1.0
                Width of the bracket.
            angleA, angleB : float, default: 0 degrees
                Orientation of the bracket, as a counterclockwise angle.
                0 degrees means perpendicular to the line.
            """
            super().__init__(widthA=widthA, lengthA=0, angleA=angleA, widthB=widthB, lengthB=0, angleB=angleB)

    @_register_style(_style_list, name=']->')
    class BracketCurve(_Curve):
        """
        An arrow with an outward square bracket at its start and a head at
        the end.
        """
        arrow = ']->'

        def __init__(self, widthA=1.0, lengthA=0.2, angleA=None):
            """
            Parameters
            ----------
            widthA : float, default: 1.0
                Width of the bracket.
            lengthA : float, default: 0.2
                Length of the bracket.
            angleA : float, default: 0 degrees
                Orientation of the bracket, as a counterclockwise angle.
                0 degrees means perpendicular to the line.
            """
            super().__init__(widthA=widthA, lengthA=lengthA, angleA=angleA)

    @_register_style(_style_list, name='<-[')
    class CurveBracket(_Curve):
        """
        An arrow with an outward square bracket at its end and a head at
        the start.
        """
        arrow = '<-['

        def __init__(self, widthB=1.0, lengthB=0.2, angleB=None):
            """
            Parameters
            ----------
            widthB : float, default: 1.0
                Width of the bracket.
            lengthB : float, default: 0.2
                Length of the bracket.
            angleB : float, default: 0 degrees
                Orientation of the bracket, as a counterclockwise angle.
                0 degrees means perpendicular to the line.
            """
            super().__init__(widthB=widthB, lengthB=lengthB, angleB=angleB)

    @_register_style(_style_list)
    class Simple(_Base):
        """A simple arrow. Only works with a quadratic Bézier curve."""

        def __init__(self, head_length=0.5, head_width=0.5, tail_width=0.2):
            """
            Parameters
            ----------
            head_length : float, default: 0.5
                Length of the arrow head.

            head_width : float, default: 0.5
                Width of the arrow head.

            tail_width : float, default: 0.2
                Width of the arrow tail.
            """
            self.head_length, self.head_width, self.tail_width = (head_length, head_width, tail_width)
            super().__init__()

        def transmute(self, path, mutation_size, linewidth):
            x0, y0, x1, y1, x2, y2 = self.ensure_quadratic_bezier(path)
            head_length = self.head_length * mutation_size
            in_f = inside_circle(x2, y2, head_length)
            arrow_path = [(x0, y0), (x1, y1), (x2, y2)]
            try:
                arrow_out, arrow_in = split_bezier_intersecting_with_closedpath(arrow_path, in_f)
            except NonIntersectingPathException:
                x0, y0 = _point_along_a_line(x2, y2, x1, y1, head_length)
                x1n, y1n = (0.5 * (x0 + x2), 0.5 * (y0 + y2))
                arrow_in = [(x0, y0), (x1n, y1n), (x2, y2)]
                arrow_out = None
            head_width = self.head_width * mutation_size
            head_left, head_right = make_wedged_bezier2(arrow_in, head_width / 2.0, wm=0.5)
            if arrow_out is not None:
                tail_width = self.tail_width * mutation_size
                tail_left, tail_right = get_parallels(arrow_out, tail_width / 2.0)
                patch_path = [(Path.MOVETO, tail_right[0]), (Path.CURVE3, tail_right[1]), (Path.CURVE3, tail_right[2]), (Path.LINETO, head_right[0]), (Path.CURVE3, head_right[1]), (Path.CURVE3, head_right[2]), (Path.CURVE3, head_left[1]), (Path.CURVE3, head_left[0]), (Path.LINETO, tail_left[2]), (Path.CURVE3, tail_left[1]), (Path.CURVE3, tail_left[0]), (Path.LINETO, tail_right[0]), (Path.CLOSEPOLY, tail_right[0])]
            else:
                patch_path = [(Path.MOVETO, head_right[0]), (Path.CURVE3, head_right[1]), (Path.CURVE3, head_right[2]), (Path.CURVE3, head_left[1]), (Path.CURVE3, head_left[0]), (Path.CLOSEPOLY, head_left[0])]
            path = Path([p for c, p in patch_path], [c for c, p in patch_path])
            return (path, True)

    @_register_style(_style_list)
    class Fancy(_Base):
        """A fancy arrow. Only works with a quadratic Bézier curve."""

        def __init__(self, head_length=0.4, head_width=0.4, tail_width=0.4):
            """
            Parameters
            ----------
            head_length : float, default: 0.4
                Length of the arrow head.

            head_width : float, default: 0.4
                Width of the arrow head.

            tail_width : float, default: 0.4
                Width of the arrow tail.
            """
            self.head_length, self.head_width, self.tail_width = (head_length, head_width, tail_width)
            super().__init__()

        def transmute(self, path, mutation_size, linewidth):
            x0, y0, x1, y1, x2, y2 = self.ensure_quadratic_bezier(path)
            head_length = self.head_length * mutation_size
            arrow_path = [(x0, y0), (x1, y1), (x2, y2)]
            in_f = inside_circle(x2, y2, head_length)
            try:
                path_out, path_in = split_bezier_intersecting_with_closedpath(arrow_path, in_f)
            except NonIntersectingPathException:
                x0, y0 = _point_along_a_line(x2, y2, x1, y1, head_length)
                x1n, y1n = (0.5 * (x0 + x2), 0.5 * (y0 + y2))
                arrow_path = [(x0, y0), (x1n, y1n), (x2, y2)]
                path_head = arrow_path
            else:
                path_head = path_in
            in_f = inside_circle(x2, y2, head_length * 0.8)
            path_out, path_in = split_bezier_intersecting_with_closedpath(arrow_path, in_f)
            path_tail = path_out
            head_width = self.head_width * mutation_size
            head_l, head_r = make_wedged_bezier2(path_head, head_width / 2.0, wm=0.6)
            tail_width = self.tail_width * mutation_size
            tail_left, tail_right = make_wedged_bezier2(path_tail, tail_width * 0.5, w1=1.0, wm=0.6, w2=0.3)
            in_f = inside_circle(x0, y0, tail_width * 0.3)
            path_in, path_out = split_bezier_intersecting_with_closedpath(arrow_path, in_f)
            tail_start = path_in[-1]
            head_right, head_left = (head_r, head_l)
            patch_path = [(Path.MOVETO, tail_start), (Path.LINETO, tail_right[0]), (Path.CURVE3, tail_right[1]), (Path.CURVE3, tail_right[2]), (Path.LINETO, head_right[0]), (Path.CURVE3, head_right[1]), (Path.CURVE3, head_right[2]), (Path.CURVE3, head_left[1]), (Path.CURVE3, head_left[0]), (Path.LINETO, tail_left[2]), (Path.CURVE3, tail_left[1]), (Path.CURVE3, tail_left[0]), (Path.LINETO, tail_start), (Path.CLOSEPOLY, tail_start)]
            path = Path([p for c, p in patch_path], [c for c, p in patch_path])
            return (path, True)

    @_register_style(_style_list)
    class Wedge(_Base):
        """
        Wedge(?) shape. Only works with a quadratic Bézier curve.  The
        start point has a width of the *tail_width* and the end point has a
        width of 0. At the middle, the width is *shrink_factor*x*tail_width*.
        """

        def __init__(self, tail_width=0.3, shrink_factor=0.5):
            """
            Parameters
            ----------
            tail_width : float, default: 0.3
                Width of the tail.

            shrink_factor : float, default: 0.5
                Fraction of the arrow width at the middle point.
            """
            self.tail_width = tail_width
            self.shrink_factor = shrink_factor
            super().__init__()

        def transmute(self, path, mutation_size, linewidth):
            x0, y0, x1, y1, x2, y2 = self.ensure_quadratic_bezier(path)
            arrow_path = [(x0, y0), (x1, y1), (x2, y2)]
            b_plus, b_minus = make_wedged_bezier2(arrow_path, self.tail_width * mutation_size / 2.0, wm=self.shrink_factor)
            patch_path = [(Path.MOVETO, b_plus[0]), (Path.CURVE3, b_plus[1]), (Path.CURVE3, b_plus[2]), (Path.LINETO, b_minus[2]), (Path.CURVE3, b_minus[1]), (Path.CURVE3, b_minus[0]), (Path.CLOSEPOLY, b_minus[0])]
            path = Path([p for c, p in patch_path], [c for c, p in patch_path])
            return (path, True)