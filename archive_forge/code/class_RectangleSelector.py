from contextlib import ExitStack
import copy
import itertools
from numbers import Integral, Number
from cycler import cycler
import numpy as np
import matplotlib as mpl
from . import (_api, _docstring, backend_tools, cbook, collections, colors,
from .lines import Line2D
from .patches import Circle, Rectangle, Ellipse, Polygon
from .transforms import TransformedPatchPath, Affine2D
@_docstring.Substitution(_RECTANGLESELECTOR_PARAMETERS_DOCSTRING.replace('__ARTIST_NAME__', 'rectangle'))
class RectangleSelector(_SelectorWidget):
    """
    Select a rectangular region of an Axes.

    For the cursor to remain responsive you must keep a reference to it.

    Press and release events triggered at the same coordinates outside the
    selection will clear the selector, except when
    ``ignore_event_outside=True``.

    %s

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import matplotlib.widgets as mwidgets
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [10, 50, 100])
    >>> def onselect(eclick, erelease):
    ...     print(eclick.xdata, eclick.ydata)
    ...     print(erelease.xdata, erelease.ydata)
    >>> props = dict(facecolor='blue', alpha=0.5)
    >>> rect = mwidgets.RectangleSelector(ax, onselect, interactive=True,
    ...                                   props=props)
    >>> fig.show()
    >>> rect.add_state('square')

    See also: :doc:`/gallery/widgets/rectangle_selector`
    """

    def __init__(self, ax, onselect, *, minspanx=0, minspany=0, useblit=False, props=None, spancoords='data', button=None, grab_range=10, handle_props=None, interactive=False, state_modifier_keys=None, drag_from_anywhere=False, ignore_event_outside=False, use_data_coordinates=False):
        super().__init__(ax, onselect, useblit=useblit, button=button, state_modifier_keys=state_modifier_keys, use_data_coordinates=use_data_coordinates)
        self._interactive = interactive
        self.drag_from_anywhere = drag_from_anywhere
        self.ignore_event_outside = ignore_event_outside
        self._rotation = 0.0
        self._aspect_ratio_correction = 1.0
        self._allow_creation = True
        if props is None:
            props = dict(facecolor='red', edgecolor='black', alpha=0.2, fill=True)
        props = {**props, 'animated': self.useblit}
        self._visible = props.pop('visible', self._visible)
        to_draw = self._init_shape(**props)
        self.ax.add_patch(to_draw)
        self._selection_artist = to_draw
        self._set_aspect_ratio_correction()
        self.minspanx = minspanx
        self.minspany = minspany
        _api.check_in_list(['data', 'pixels'], spancoords=spancoords)
        self.spancoords = spancoords
        self.grab_range = grab_range
        if self._interactive:
            self._handle_props = {'markeredgecolor': (props or {}).get('edgecolor', 'black'), **cbook.normalize_kwargs(handle_props, Line2D)}
            self._corner_order = ['SW', 'SE', 'NE', 'NW']
            xc, yc = self.corners
            self._corner_handles = ToolHandles(self.ax, xc, yc, marker_props=self._handle_props, useblit=self.useblit)
            self._edge_order = ['W', 'S', 'E', 'N']
            xe, ye = self.edge_centers
            self._edge_handles = ToolHandles(self.ax, xe, ye, marker='s', marker_props=self._handle_props, useblit=self.useblit)
            xc, yc = self.center
            self._center_handle = ToolHandles(self.ax, [xc], [yc], marker='s', marker_props=self._handle_props, useblit=self.useblit)
            self._active_handle = None
        self._extents_on_press = None

    @property
    def _handles_artists(self):
        return (*self._center_handle.artists, *self._corner_handles.artists, *self._edge_handles.artists)

    def _init_shape(self, **props):
        return Rectangle((0, 0), 0, 1, visible=False, rotation_point='center', **props)

    def _press(self, event):
        """Button press event handler."""
        if self._interactive and self._selection_artist.get_visible():
            self._set_active_handle(event)
        else:
            self._active_handle = None
        if (self._active_handle is None or not self._interactive) and self._allow_creation:
            self.update()
        if self._active_handle is None and (not self.ignore_event_outside) and self._allow_creation:
            x, y = self._get_data_coords(event)
            self._visible = False
            self.extents = (x, x, y, y)
            self._visible = True
        else:
            self.set_visible(True)
        self._extents_on_press = self.extents
        self._rotation_on_press = self._rotation
        self._set_aspect_ratio_correction()
        return False

    def _release(self, event):
        """Button release event handler."""
        if not self._interactive:
            self._selection_artist.set_visible(False)
        if self._active_handle is None and self._selection_completed and self.ignore_event_outside:
            return
        x0, x1, y0, y1 = self.extents
        self._eventpress.xdata = x0
        self._eventpress.ydata = y0
        xy0 = self.ax.transData.transform([x0, y0])
        self._eventpress.x, self._eventpress.y = xy0
        self._eventrelease.xdata = x1
        self._eventrelease.ydata = y1
        xy1 = self.ax.transData.transform([x1, y1])
        self._eventrelease.x, self._eventrelease.y = xy1
        if self.spancoords == 'data':
            spanx = abs(self._eventpress.xdata - self._eventrelease.xdata)
            spany = abs(self._eventpress.ydata - self._eventrelease.ydata)
        elif self.spancoords == 'pixels':
            spanx = abs(self._eventpress.x - self._eventrelease.x)
            spany = abs(self._eventpress.y - self._eventrelease.y)
        else:
            _api.check_in_list(['data', 'pixels'], spancoords=self.spancoords)
        if spanx <= self.minspanx or spany <= self.minspany:
            if self._selection_completed:
                self.onselect(self._eventpress, self._eventrelease)
            self._clear_without_update()
        else:
            self.onselect(self._eventpress, self._eventrelease)
            self._selection_completed = True
        self.update()
        self._active_handle = None
        self._extents_on_press = None
        return False

    def _onmove(self, event):
        """
        Motion notify event handler.

        This can do one of four things:
        - Translate
        - Rotate
        - Re-size
        - Continue the creation of a new shape
        """
        eventpress = self._eventpress
        state = self._state
        rotate = 'rotate' in state and self._active_handle in self._corner_order
        move = self._active_handle == 'C'
        resize = self._active_handle and (not move)
        xdata, ydata = self._get_data_coords(event)
        if resize:
            inv_tr = self._get_rotation_transform().inverted()
            xdata, ydata = inv_tr.transform([xdata, ydata])
            eventpress.xdata, eventpress.ydata = inv_tr.transform((eventpress.xdata, eventpress.ydata))
        dx = xdata - eventpress.xdata
        dy = ydata - eventpress.ydata
        refmax = None
        if self._use_data_coordinates:
            refx, refy = (dx, dy)
        else:
            refx = event.x - eventpress.x
            refy = event.y - eventpress.y
        x0, x1, y0, y1 = self._extents_on_press
        if rotate:
            a = (eventpress.xdata, eventpress.ydata)
            b = self.center
            c = (xdata, ydata)
            angle = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            self.rotation = np.rad2deg(self._rotation_on_press + angle)
        elif resize:
            size_on_press = [x1 - x0, y1 - y0]
            center = (x0 + size_on_press[0] / 2, y0 + size_on_press[1] / 2)
            if 'center' in state:
                if 'square' in state:
                    if self._active_handle in self._corner_order:
                        refmax = max(refx, refy, key=abs)
                    if self._active_handle in ['E', 'W'] or refmax == refx:
                        hw = xdata - center[0]
                        hh = hw / self._aspect_ratio_correction
                    else:
                        hh = ydata - center[1]
                        hw = hh * self._aspect_ratio_correction
                else:
                    hw = size_on_press[0] / 2
                    hh = size_on_press[1] / 2
                    if self._active_handle in ['E', 'W'] + self._corner_order:
                        hw = abs(xdata - center[0])
                    if self._active_handle in ['N', 'S'] + self._corner_order:
                        hh = abs(ydata - center[1])
                x0, x1, y0, y1 = (center[0] - hw, center[0] + hw, center[1] - hh, center[1] + hh)
            else:
                if 'W' in self._active_handle:
                    x0 = x1
                if 'S' in self._active_handle:
                    y0 = y1
                if self._active_handle in ['E', 'W'] + self._corner_order:
                    x1 = xdata
                if self._active_handle in ['N', 'S'] + self._corner_order:
                    y1 = ydata
                if 'square' in state:
                    if self._active_handle in self._corner_order:
                        refmax = max(refx, refy, key=abs)
                    if self._active_handle in ['E', 'W'] or refmax == refx:
                        sign = np.sign(ydata - y0)
                        y1 = y0 + sign * abs(x1 - x0) / self._aspect_ratio_correction
                    else:
                        sign = np.sign(xdata - x0)
                        x1 = x0 + sign * abs(y1 - y0) * self._aspect_ratio_correction
        elif move:
            x0, x1, y0, y1 = self._extents_on_press
            dx = xdata - eventpress.xdata
            dy = ydata - eventpress.ydata
            x0 += dx
            x1 += dx
            y0 += dy
            y1 += dy
        else:
            self._rotation = 0
            if self.ignore_event_outside and self._selection_completed or not self._allow_creation:
                return
            center = [eventpress.xdata, eventpress.ydata]
            dx = (xdata - center[0]) / 2
            dy = (ydata - center[1]) / 2
            if 'square' in state:
                refmax = max(refx, refy, key=abs)
                if refmax == refx:
                    dy = np.sign(dy) * abs(dx) / self._aspect_ratio_correction
                else:
                    dx = np.sign(dx) * abs(dy) * self._aspect_ratio_correction
            if 'center' in state:
                dx *= 2
                dy *= 2
            else:
                center[0] += dx
                center[1] += dy
            x0, x1, y0, y1 = (center[0] - dx, center[0] + dx, center[1] - dy, center[1] + dy)
        self.extents = (x0, x1, y0, y1)

    @property
    def _rect_bbox(self):
        return self._selection_artist.get_bbox().bounds

    def _set_aspect_ratio_correction(self):
        aspect_ratio = self.ax._get_aspect_ratio()
        self._selection_artist._aspect_ratio_correction = aspect_ratio
        if self._use_data_coordinates:
            self._aspect_ratio_correction = 1
        else:
            self._aspect_ratio_correction = aspect_ratio

    def _get_rotation_transform(self):
        aspect_ratio = self.ax._get_aspect_ratio()
        return Affine2D().translate(-self.center[0], -self.center[1]).scale(1, aspect_ratio).rotate(self._rotation).scale(1, 1 / aspect_ratio).translate(*self.center)

    @property
    def corners(self):
        """
        Corners of rectangle in data coordinates from lower left,
        moving clockwise.
        """
        x0, y0, width, height = self._rect_bbox
        xc = (x0, x0 + width, x0 + width, x0)
        yc = (y0, y0, y0 + height, y0 + height)
        transform = self._get_rotation_transform()
        coords = transform.transform(np.array([xc, yc]).T).T
        return (coords[0], coords[1])

    @property
    def edge_centers(self):
        """
        Midpoint of rectangle edges in data coordinates from left,
        moving anti-clockwise.
        """
        x0, y0, width, height = self._rect_bbox
        w = width / 2.0
        h = height / 2.0
        xe = (x0, x0 + w, x0 + width, x0 + w)
        ye = (y0 + h, y0, y0 + h, y0 + height)
        transform = self._get_rotation_transform()
        coords = transform.transform(np.array([xe, ye]).T).T
        return (coords[0], coords[1])

    @property
    def center(self):
        """Center of rectangle in data coordinates."""
        x0, y0, width, height = self._rect_bbox
        return (x0 + width / 2.0, y0 + height / 2.0)

    @property
    def extents(self):
        """
        Return (xmin, xmax, ymin, ymax) in data coordinates as defined by the
        bounding box before rotation.
        """
        x0, y0, width, height = self._rect_bbox
        xmin, xmax = sorted([x0, x0 + width])
        ymin, ymax = sorted([y0, y0 + height])
        return (xmin, xmax, ymin, ymax)

    @extents.setter
    def extents(self, extents):
        self._draw_shape(extents)
        if self._interactive:
            self._corner_handles.set_data(*self.corners)
            self._edge_handles.set_data(*self.edge_centers)
            x, y = self.center
            self._center_handle.set_data([x], [y])
        self.set_visible(self._visible)
        self.update()

    @property
    def rotation(self):
        """
        Rotation in degree in interval [-45°, 45°]. The rotation is limited in
        range to keep the implementation simple.
        """
        return np.rad2deg(self._rotation)

    @rotation.setter
    def rotation(self, value):
        if -45 <= value and value <= 45:
            self._rotation = np.deg2rad(value)
            self.extents = self.extents

    def _draw_shape(self, extents):
        x0, x1, y0, y1 = extents
        xmin, xmax = sorted([x0, x1])
        ymin, ymax = sorted([y0, y1])
        xlim = sorted(self.ax.get_xlim())
        ylim = sorted(self.ax.get_ylim())
        xmin = max(xlim[0], xmin)
        ymin = max(ylim[0], ymin)
        xmax = min(xmax, xlim[1])
        ymax = min(ymax, ylim[1])
        self._selection_artist.set_x(xmin)
        self._selection_artist.set_y(ymin)
        self._selection_artist.set_width(xmax - xmin)
        self._selection_artist.set_height(ymax - ymin)
        self._selection_artist.set_angle(self.rotation)

    def _set_active_handle(self, event):
        """Set active handle based on the location of the mouse event."""
        c_idx, c_dist = self._corner_handles.closest(event.x, event.y)
        e_idx, e_dist = self._edge_handles.closest(event.x, event.y)
        m_idx, m_dist = self._center_handle.closest(event.x, event.y)
        if 'move' in self._state:
            self._active_handle = 'C'
        elif m_dist < self.grab_range * 2:
            self._active_handle = 'C'
        elif c_dist > self.grab_range and e_dist > self.grab_range:
            if self.drag_from_anywhere and self._contains(event):
                self._active_handle = 'C'
            else:
                self._active_handle = None
                return
        elif c_dist < e_dist:
            self._active_handle = self._corner_order[c_idx]
        else:
            self._active_handle = self._edge_order[e_idx]

    def _contains(self, event):
        """Return True if event is within the patch."""
        return self._selection_artist.contains(event, radius=0)[0]

    @property
    def geometry(self):
        """
        Return an array of shape (2, 5) containing the
        x (``RectangleSelector.geometry[1, :]``) and
        y (``RectangleSelector.geometry[0, :]``) data coordinates of the four
        corners of the rectangle starting and ending in the top left corner.
        """
        if hasattr(self._selection_artist, 'get_verts'):
            xfm = self.ax.transData.inverted()
            y, x = xfm.transform(self._selection_artist.get_verts()).T
            return np.array([x, y])
        else:
            return np.array(self._selection_artist.get_data())