import copy
from collections.abc import Sized
import numpy as np
import matplotlib as mpl
from . import _api, cbook
from .path import Path
from .transforms import IdentityTransform, Affine2D
from ._enums import JoinStyle, CapStyle
class MarkerStyle:
    """
    A class representing marker types.

    Instances are immutable. If you need to change anything, create a new
    instance.

    Attributes
    ----------
    markers : dict
        All known markers.
    filled_markers : tuple
        All known filled markers. This is a subset of *markers*.
    fillstyles : tuple
        The supported fillstyles.
    """
    markers = {'.': 'point', ',': 'pixel', 'o': 'circle', 'v': 'triangle_down', '^': 'triangle_up', '<': 'triangle_left', '>': 'triangle_right', '1': 'tri_down', '2': 'tri_up', '3': 'tri_left', '4': 'tri_right', '8': 'octagon', 's': 'square', 'p': 'pentagon', '*': 'star', 'h': 'hexagon1', 'H': 'hexagon2', '+': 'plus', 'x': 'x', 'D': 'diamond', 'd': 'thin_diamond', '|': 'vline', '_': 'hline', 'P': 'plus_filled', 'X': 'x_filled', TICKLEFT: 'tickleft', TICKRIGHT: 'tickright', TICKUP: 'tickup', TICKDOWN: 'tickdown', CARETLEFT: 'caretleft', CARETRIGHT: 'caretright', CARETUP: 'caretup', CARETDOWN: 'caretdown', CARETLEFTBASE: 'caretleftbase', CARETRIGHTBASE: 'caretrightbase', CARETUPBASE: 'caretupbase', CARETDOWNBASE: 'caretdownbase', 'None': 'nothing', 'none': 'nothing', ' ': 'nothing', '': 'nothing'}
    filled_markers = ('.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
    fillstyles = ('full', 'left', 'right', 'bottom', 'top', 'none')
    _half_fillstyles = ('left', 'right', 'bottom', 'top')

    def __init__(self, marker, fillstyle=None, transform=None, capstyle=None, joinstyle=None):
        """
        Parameters
        ----------
        marker : str, array-like, Path, MarkerStyle, or None
            - Another instance of *MarkerStyle* copies the details of that
              ``marker``.
            - *None* means no marker.  This is the deprecated default.
            - For other possible marker values, see the module docstring
              `matplotlib.markers`.

        fillstyle : str, default: :rc:`markers.fillstyle`
            One of 'full', 'left', 'right', 'bottom', 'top', 'none'.

        transform : transforms.Transform, default: None
            Transform that will be combined with the native transform of the
            marker.

        capstyle : `.CapStyle` or %(CapStyle)s, default: None
            Cap style that will override the default cap style of the marker.

        joinstyle : `.JoinStyle` or %(JoinStyle)s, default: None
            Join style that will override the default join style of the marker.
        """
        self._marker_function = None
        self._user_transform = transform
        self._user_capstyle = CapStyle(capstyle) if capstyle is not None else None
        self._user_joinstyle = JoinStyle(joinstyle) if joinstyle is not None else None
        self._set_fillstyle(fillstyle)
        self._set_marker(marker)

    def _recache(self):
        if self._marker_function is None:
            return
        self._path = _empty_path
        self._transform = IdentityTransform()
        self._alt_path = None
        self._alt_transform = None
        self._snap_threshold = None
        self._joinstyle = JoinStyle.round
        self._capstyle = self._user_capstyle or CapStyle.butt
        self._filled = self._fillstyle != 'none'
        self._marker_function()

    def __bool__(self):
        return bool(len(self._path.vertices))

    def is_filled(self):
        return self._filled

    def get_fillstyle(self):
        return self._fillstyle

    def _set_fillstyle(self, fillstyle):
        """
        Set the fillstyle.

        Parameters
        ----------
        fillstyle : {'full', 'left', 'right', 'bottom', 'top', 'none'}
            The part of the marker surface that is colored with
            markerfacecolor.
        """
        if fillstyle is None:
            fillstyle = mpl.rcParams['markers.fillstyle']
        _api.check_in_list(self.fillstyles, fillstyle=fillstyle)
        self._fillstyle = fillstyle

    def get_joinstyle(self):
        return self._joinstyle.name

    def get_capstyle(self):
        return self._capstyle.name

    def get_marker(self):
        return self._marker

    def _set_marker(self, marker):
        """
        Set the marker.

        Parameters
        ----------
        marker : str, array-like, Path, MarkerStyle, or None, default: None
            - Another instance of *MarkerStyle* copies the details of that
              ``marker``.
            - *None* means no marker.
            - For other possible marker values see the module docstring
              `matplotlib.markers`.
        """
        if isinstance(marker, str) and cbook.is_math_text(marker):
            self._marker_function = self._set_mathtext_path
        elif isinstance(marker, (int, str)) and marker in self.markers:
            self._marker_function = getattr(self, '_set_' + self.markers[marker])
        elif isinstance(marker, np.ndarray) and marker.ndim == 2 and (marker.shape[1] == 2):
            self._marker_function = self._set_vertices
        elif isinstance(marker, Path):
            self._marker_function = self._set_path_marker
        elif isinstance(marker, Sized) and len(marker) in (2, 3) and (marker[1] in (0, 1, 2)):
            self._marker_function = self._set_tuple_marker
        elif isinstance(marker, MarkerStyle):
            self.__dict__ = copy.deepcopy(marker.__dict__)
        else:
            try:
                Path(marker)
                self._marker_function = self._set_vertices
            except ValueError as err:
                raise ValueError(f'Unrecognized marker style {marker!r}') from err
        if not isinstance(marker, MarkerStyle):
            self._marker = marker
            self._recache()

    def get_path(self):
        """
        Return a `.Path` for the primary part of the marker.

        For unfilled markers this is the whole marker, for filled markers,
        this is the area to be drawn with *markerfacecolor*.
        """
        return self._path

    def get_transform(self):
        """
        Return the transform to be applied to the `.Path` from
        `MarkerStyle.get_path()`.
        """
        if self._user_transform is None:
            return self._transform.frozen()
        else:
            return (self._transform + self._user_transform).frozen()

    def get_alt_path(self):
        """
        Return a `.Path` for the alternate part of the marker.

        For unfilled markers, this is *None*; for filled markers, this is the
        area to be drawn with *markerfacecoloralt*.
        """
        return self._alt_path

    def get_alt_transform(self):
        """
        Return the transform to be applied to the `.Path` from
        `MarkerStyle.get_alt_path()`.
        """
        if self._user_transform is None:
            return self._alt_transform.frozen()
        else:
            return (self._alt_transform + self._user_transform).frozen()

    def get_snap_threshold(self):
        return self._snap_threshold

    def get_user_transform(self):
        """Return user supplied part of marker transform."""
        if self._user_transform is not None:
            return self._user_transform.frozen()

    def transformed(self, transform: Affine2D):
        """
        Return a new version of this marker with the transform applied.

        Parameters
        ----------
        transform : `~matplotlib.transforms.Affine2D`, default: None
            Transform will be combined with current user supplied transform.
        """
        new_marker = MarkerStyle(self)
        if new_marker._user_transform is not None:
            new_marker._user_transform += transform
        else:
            new_marker._user_transform = transform
        return new_marker

    def rotated(self, *, deg=None, rad=None):
        """
        Return a new version of this marker rotated by specified angle.

        Parameters
        ----------
        deg : float, default: None
            Rotation angle in degrees.

        rad : float, default: None
            Rotation angle in radians.

        .. note:: You must specify exactly one of deg or rad.
        """
        if deg is None and rad is None:
            raise ValueError('One of deg or rad is required')
        if deg is not None and rad is not None:
            raise ValueError('Only one of deg and rad can be supplied')
        new_marker = MarkerStyle(self)
        if new_marker._user_transform is None:
            new_marker._user_transform = Affine2D()
        if deg is not None:
            new_marker._user_transform.rotate_deg(deg)
        if rad is not None:
            new_marker._user_transform.rotate(rad)
        return new_marker

    def scaled(self, sx, sy=None):
        """
        Return new marker scaled by specified scale factors.

        If *sy* is None, the same scale is applied in both the *x*- and
        *y*-directions.

        Parameters
        ----------
        sx : float
            *X*-direction scaling factor.
        sy : float, default: None
            *Y*-direction scaling factor.
        """
        if sy is None:
            sy = sx
        new_marker = MarkerStyle(self)
        _transform = new_marker._user_transform or Affine2D()
        new_marker._user_transform = _transform.scale(sx, sy)
        return new_marker

    def _set_nothing(self):
        self._filled = False

    def _set_custom_marker(self, path):
        rescale = np.max(np.abs(path.vertices))
        self._transform = Affine2D().scale(0.5 / rescale)
        self._path = path

    def _set_path_marker(self):
        self._set_custom_marker(self._marker)

    def _set_vertices(self):
        self._set_custom_marker(Path(self._marker))

    def _set_tuple_marker(self):
        marker = self._marker
        if len(marker) == 2:
            numsides, rotation = (marker[0], 0.0)
        elif len(marker) == 3:
            numsides, rotation = (marker[0], marker[2])
        symstyle = marker[1]
        if symstyle == 0:
            self._path = Path.unit_regular_polygon(numsides)
            self._joinstyle = self._user_joinstyle or JoinStyle.miter
        elif symstyle == 1:
            self._path = Path.unit_regular_star(numsides)
            self._joinstyle = self._user_joinstyle or JoinStyle.bevel
        elif symstyle == 2:
            self._path = Path.unit_regular_asterisk(numsides)
            self._filled = False
            self._joinstyle = self._user_joinstyle or JoinStyle.bevel
        else:
            raise ValueError(f'Unexpected tuple marker: {marker}')
        self._transform = Affine2D().scale(0.5).rotate_deg(rotation)

    def _set_mathtext_path(self):
        """
        Draw mathtext markers '$...$' using `.TextPath` object.

        Submitted by tcb
        """
        from matplotlib.text import TextPath
        text = TextPath(xy=(0, 0), s=self.get_marker(), usetex=mpl.rcParams['text.usetex'])
        if len(text.vertices) == 0:
            return
        bbox = text.get_extents()
        max_dim = max(bbox.width, bbox.height)
        self._transform = Affine2D().translate(-bbox.xmin + 0.5 * -bbox.width, -bbox.ymin + 0.5 * -bbox.height).scale(1.0 / max_dim)
        self._path = text
        self._snap = False

    def _half_fill(self):
        return self.get_fillstyle() in self._half_fillstyles

    def _set_circle(self, size=1.0):
        self._transform = Affine2D().scale(0.5 * size)
        self._snap_threshold = np.inf
        if not self._half_fill():
            self._path = Path.unit_circle()
        else:
            self._path = self._alt_path = Path.unit_circle_righthalf()
            fs = self.get_fillstyle()
            self._transform.rotate_deg({'right': 0, 'top': 90, 'left': 180, 'bottom': 270}[fs])
            self._alt_transform = self._transform.frozen().rotate_deg(180.0)

    def _set_point(self):
        self._set_circle(size=0.5)

    def _set_pixel(self):
        self._path = Path.unit_rectangle()
        self._transform = Affine2D().translate(-0.49999, -0.49999)
        self._snap_threshold = None
    _triangle_path = Path._create_closed([[0, 1], [-1, -1], [1, -1]])
    _triangle_path_u = Path._create_closed([[0, 1], [-3 / 5, -1 / 5], [3 / 5, -1 / 5]])
    _triangle_path_d = Path._create_closed([[-3 / 5, -1 / 5], [3 / 5, -1 / 5], [1, -1], [-1, -1]])
    _triangle_path_l = Path._create_closed([[0, 1], [0, -1], [-1, -1]])
    _triangle_path_r = Path._create_closed([[0, 1], [0, -1], [1, -1]])

    def _set_triangle(self, rot, skip):
        self._transform = Affine2D().scale(0.5).rotate_deg(rot)
        self._snap_threshold = 5.0
        if not self._half_fill():
            self._path = self._triangle_path
        else:
            mpaths = [self._triangle_path_u, self._triangle_path_l, self._triangle_path_d, self._triangle_path_r]
            fs = self.get_fillstyle()
            if fs == 'top':
                self._path = mpaths[(0 + skip) % 4]
                self._alt_path = mpaths[(2 + skip) % 4]
            elif fs == 'bottom':
                self._path = mpaths[(2 + skip) % 4]
                self._alt_path = mpaths[(0 + skip) % 4]
            elif fs == 'left':
                self._path = mpaths[(1 + skip) % 4]
                self._alt_path = mpaths[(3 + skip) % 4]
            else:
                self._path = mpaths[(3 + skip) % 4]
                self._alt_path = mpaths[(1 + skip) % 4]
            self._alt_transform = self._transform
        self._joinstyle = self._user_joinstyle or JoinStyle.miter

    def _set_triangle_up(self):
        return self._set_triangle(0.0, 0)

    def _set_triangle_down(self):
        return self._set_triangle(180.0, 2)

    def _set_triangle_left(self):
        return self._set_triangle(90.0, 3)

    def _set_triangle_right(self):
        return self._set_triangle(270.0, 1)

    def _set_square(self):
        self._transform = Affine2D().translate(-0.5, -0.5)
        self._snap_threshold = 2.0
        if not self._half_fill():
            self._path = Path.unit_rectangle()
        else:
            self._path = Path([[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [0.0, 0.5], [0.0, 0.0]])
            self._alt_path = Path([[0.0, 0.5], [1.0, 0.5], [1.0, 1.0], [0.0, 1.0], [0.0, 0.5]])
            fs = self.get_fillstyle()
            rotate = {'bottom': 0, 'right': 90, 'top': 180, 'left': 270}[fs]
            self._transform.rotate_deg(rotate)
            self._alt_transform = self._transform
        self._joinstyle = self._user_joinstyle or JoinStyle.miter

    def _set_diamond(self):
        self._transform = Affine2D().translate(-0.5, -0.5).rotate_deg(45)
        self._snap_threshold = 5.0
        if not self._half_fill():
            self._path = Path.unit_rectangle()
        else:
            self._path = Path([[0, 0], [1, 0], [1, 1], [0, 0]])
            self._alt_path = Path([[0, 0], [0, 1], [1, 1], [0, 0]])
            fs = self.get_fillstyle()
            rotate = {'right': 0, 'top': 90, 'left': 180, 'bottom': 270}[fs]
            self._transform.rotate_deg(rotate)
            self._alt_transform = self._transform
        self._joinstyle = self._user_joinstyle or JoinStyle.miter

    def _set_thin_diamond(self):
        self._set_diamond()
        self._transform.scale(0.6, 1.0)

    def _set_pentagon(self):
        self._transform = Affine2D().scale(0.5)
        self._snap_threshold = 5.0
        polypath = Path.unit_regular_polygon(5)
        if not self._half_fill():
            self._path = polypath
        else:
            verts = polypath.vertices
            y = (1 + np.sqrt(5)) / 4.0
            top = Path(verts[[0, 1, 4, 0]])
            bottom = Path(verts[[1, 2, 3, 4, 1]])
            left = Path([verts[0], verts[1], verts[2], [0, -y], verts[0]])
            right = Path([verts[0], verts[4], verts[3], [0, -y], verts[0]])
            self._path, self._alt_path = {'top': (top, bottom), 'bottom': (bottom, top), 'left': (left, right), 'right': (right, left)}[self.get_fillstyle()]
            self._alt_transform = self._transform
        self._joinstyle = self._user_joinstyle or JoinStyle.miter

    def _set_star(self):
        self._transform = Affine2D().scale(0.5)
        self._snap_threshold = 5.0
        polypath = Path.unit_regular_star(5, innerCircle=0.381966)
        if not self._half_fill():
            self._path = polypath
        else:
            verts = polypath.vertices
            top = Path(np.concatenate([verts[0:4], verts[7:10], verts[0:1]]))
            bottom = Path(np.concatenate([verts[3:8], verts[3:4]]))
            left = Path(np.concatenate([verts[0:6], verts[0:1]]))
            right = Path(np.concatenate([verts[0:1], verts[5:10], verts[0:1]]))
            self._path, self._alt_path = {'top': (top, bottom), 'bottom': (bottom, top), 'left': (left, right), 'right': (right, left)}[self.get_fillstyle()]
            self._alt_transform = self._transform
        self._joinstyle = self._user_joinstyle or JoinStyle.bevel

    def _set_hexagon1(self):
        self._transform = Affine2D().scale(0.5)
        self._snap_threshold = None
        polypath = Path.unit_regular_polygon(6)
        if not self._half_fill():
            self._path = polypath
        else:
            verts = polypath.vertices
            x = np.abs(np.cos(5 * np.pi / 6.0))
            top = Path(np.concatenate([[(-x, 0)], verts[[1, 0, 5]], [(x, 0)]]))
            bottom = Path(np.concatenate([[(-x, 0)], verts[2:5], [(x, 0)]]))
            left = Path(verts[0:4])
            right = Path(verts[[0, 5, 4, 3]])
            self._path, self._alt_path = {'top': (top, bottom), 'bottom': (bottom, top), 'left': (left, right), 'right': (right, left)}[self.get_fillstyle()]
            self._alt_transform = self._transform
        self._joinstyle = self._user_joinstyle or JoinStyle.miter

    def _set_hexagon2(self):
        self._transform = Affine2D().scale(0.5).rotate_deg(30)
        self._snap_threshold = None
        polypath = Path.unit_regular_polygon(6)
        if not self._half_fill():
            self._path = polypath
        else:
            verts = polypath.vertices
            x, y = (np.sqrt(3) / 4, 3 / 4.0)
            top = Path(verts[[1, 0, 5, 4, 1]])
            bottom = Path(verts[1:5])
            left = Path(np.concatenate([[(x, y)], verts[:3], [(-x, -y), (x, y)]]))
            right = Path(np.concatenate([[(x, y)], verts[5:2:-1], [(-x, -y)]]))
            self._path, self._alt_path = {'top': (top, bottom), 'bottom': (bottom, top), 'left': (left, right), 'right': (right, left)}[self.get_fillstyle()]
            self._alt_transform = self._transform
        self._joinstyle = self._user_joinstyle or JoinStyle.miter

    def _set_octagon(self):
        self._transform = Affine2D().scale(0.5)
        self._snap_threshold = 5.0
        polypath = Path.unit_regular_polygon(8)
        if not self._half_fill():
            self._transform.rotate_deg(22.5)
            self._path = polypath
        else:
            x = np.sqrt(2.0) / 4.0
            self._path = self._alt_path = Path([[0, -1], [0, 1], [-x, 1], [-1, x], [-1, -x], [-x, -1], [0, -1]])
            fs = self.get_fillstyle()
            self._transform.rotate_deg({'left': 0, 'bottom': 90, 'right': 180, 'top': 270}[fs])
            self._alt_transform = self._transform.frozen().rotate_deg(180.0)
        self._joinstyle = self._user_joinstyle or JoinStyle.miter
    _line_marker_path = Path([[0.0, -1.0], [0.0, 1.0]])

    def _set_vline(self):
        self._transform = Affine2D().scale(0.5)
        self._snap_threshold = 1.0
        self._filled = False
        self._path = self._line_marker_path

    def _set_hline(self):
        self._set_vline()
        self._transform = self._transform.rotate_deg(90)
    _tickhoriz_path = Path([[0.0, 0.0], [1.0, 0.0]])

    def _set_tickleft(self):
        self._transform = Affine2D().scale(-1.0, 1.0)
        self._snap_threshold = 1.0
        self._filled = False
        self._path = self._tickhoriz_path

    def _set_tickright(self):
        self._transform = Affine2D().scale(1.0, 1.0)
        self._snap_threshold = 1.0
        self._filled = False
        self._path = self._tickhoriz_path
    _tickvert_path = Path([[-0.0, 0.0], [-0.0, 1.0]])

    def _set_tickup(self):
        self._transform = Affine2D().scale(1.0, 1.0)
        self._snap_threshold = 1.0
        self._filled = False
        self._path = self._tickvert_path

    def _set_tickdown(self):
        self._transform = Affine2D().scale(1.0, -1.0)
        self._snap_threshold = 1.0
        self._filled = False
        self._path = self._tickvert_path
    _tri_path = Path([[0.0, 0.0], [0.0, -1.0], [0.0, 0.0], [0.8, 0.5], [0.0, 0.0], [-0.8, 0.5]], [Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO])

    def _set_tri_down(self):
        self._transform = Affine2D().scale(0.5)
        self._snap_threshold = 5.0
        self._filled = False
        self._path = self._tri_path

    def _set_tri_up(self):
        self._set_tri_down()
        self._transform = self._transform.rotate_deg(180)

    def _set_tri_left(self):
        self._set_tri_down()
        self._transform = self._transform.rotate_deg(270)

    def _set_tri_right(self):
        self._set_tri_down()
        self._transform = self._transform.rotate_deg(90)
    _caret_path = Path([[-1.0, 1.5], [0.0, 0.0], [1.0, 1.5]])

    def _set_caretdown(self):
        self._transform = Affine2D().scale(0.5)
        self._snap_threshold = 3.0
        self._filled = False
        self._path = self._caret_path
        self._joinstyle = self._user_joinstyle or JoinStyle.miter

    def _set_caretup(self):
        self._set_caretdown()
        self._transform = self._transform.rotate_deg(180)

    def _set_caretleft(self):
        self._set_caretdown()
        self._transform = self._transform.rotate_deg(270)

    def _set_caretright(self):
        self._set_caretdown()
        self._transform = self._transform.rotate_deg(90)
    _caret_path_base = Path([[-1.0, 0.0], [0.0, -1.5], [1.0, 0]])

    def _set_caretdownbase(self):
        self._set_caretdown()
        self._path = self._caret_path_base

    def _set_caretupbase(self):
        self._set_caretdownbase()
        self._transform = self._transform.rotate_deg(180)

    def _set_caretleftbase(self):
        self._set_caretdownbase()
        self._transform = self._transform.rotate_deg(270)

    def _set_caretrightbase(self):
        self._set_caretdownbase()
        self._transform = self._transform.rotate_deg(90)
    _plus_path = Path([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]], [Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO])

    def _set_plus(self):
        self._transform = Affine2D().scale(0.5)
        self._snap_threshold = 1.0
        self._filled = False
        self._path = self._plus_path
    _x_path = Path([[-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]], [Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO])

    def _set_x(self):
        self._transform = Affine2D().scale(0.5)
        self._snap_threshold = 3.0
        self._filled = False
        self._path = self._x_path
    _plus_filled_path = Path._create_closed(np.array([(-1, -3), (+1, -3), (+1, -1), (+3, -1), (+3, +1), (+1, +1), (+1, +3), (-1, +3), (-1, +1), (-3, +1), (-3, -1), (-1, -1)]) / 6)
    _plus_filled_path_t = Path._create_closed(np.array([(+3, 0), (+3, +1), (+1, +1), (+1, +3), (-1, +3), (-1, +1), (-3, +1), (-3, 0)]) / 6)

    def _set_plus_filled(self):
        self._transform = Affine2D()
        self._snap_threshold = 5.0
        self._joinstyle = self._user_joinstyle or JoinStyle.miter
        if not self._half_fill():
            self._path = self._plus_filled_path
        else:
            self._path = self._alt_path = self._plus_filled_path_t
            fs = self.get_fillstyle()
            self._transform.rotate_deg({'top': 0, 'left': 90, 'bottom': 180, 'right': 270}[fs])
            self._alt_transform = self._transform.frozen().rotate_deg(180)
    _x_filled_path = Path._create_closed(np.array([(-1, -2), (0, -1), (+1, -2), (+2, -1), (+1, 0), (+2, +1), (+1, +2), (0, +1), (-1, +2), (-2, +1), (-1, 0), (-2, -1)]) / 4)
    _x_filled_path_t = Path._create_closed(np.array([(+1, 0), (+2, +1), (+1, +2), (0, +1), (-1, +2), (-2, +1), (-1, 0)]) / 4)

    def _set_x_filled(self):
        self._transform = Affine2D()
        self._snap_threshold = 5.0
        self._joinstyle = self._user_joinstyle or JoinStyle.miter
        if not self._half_fill():
            self._path = self._x_filled_path
        else:
            self._path = self._alt_path = self._x_filled_path_t
            fs = self.get_fillstyle()
            self._transform.rotate_deg({'top': 0, 'left': 90, 'bottom': 180, 'right': 270}[fs])
            self._alt_transform = self._transform.frozen().rotate_deg(180)