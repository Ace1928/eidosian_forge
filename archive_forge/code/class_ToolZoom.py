import enum
import functools
import re
import time
from types import SimpleNamespace
import uuid
from weakref import WeakKeyDictionary
import numpy as np
import matplotlib as mpl
from matplotlib._pylab_helpers import Gcf
from matplotlib import _api, cbook
class ToolZoom(ZoomPanBase):
    """A Tool for zooming using a rectangle selector."""
    description = 'Zoom to rectangle'
    image = 'zoom_to_rect'
    default_keymap = property(lambda self: mpl.rcParams['keymap.zoom'])
    cursor = cursors.SELECT_REGION
    radio_group = 'default'

    def __init__(self, *args):
        super().__init__(*args)
        self._ids_zoom = []

    def _cancel_action(self):
        for zoom_id in self._ids_zoom:
            self.figure.canvas.mpl_disconnect(zoom_id)
        self.toolmanager.trigger_tool('rubberband', self)
        self.figure.canvas.draw_idle()
        self._xypress = None
        self._button_pressed = None
        self._ids_zoom = []
        return

    def _press(self, event):
        """Callback for mouse button presses in zoom-to-rectangle mode."""
        if self._ids_zoom:
            self._cancel_action()
        if event.button == 1:
            self._button_pressed = 1
        elif event.button == 3:
            self._button_pressed = 3
        else:
            self._cancel_action()
            return
        x, y = (event.x, event.y)
        self._xypress = []
        for i, a in enumerate(self.figure.get_axes()):
            if x is not None and y is not None and a.in_axes(event) and a.get_navigate() and a.can_zoom():
                self._xypress.append((x, y, a, i, a._get_view()))
        id1 = self.figure.canvas.mpl_connect('motion_notify_event', self._mouse_move)
        id2 = self.figure.canvas.mpl_connect('key_press_event', self._switch_on_zoom_mode)
        id3 = self.figure.canvas.mpl_connect('key_release_event', self._switch_off_zoom_mode)
        self._ids_zoom = (id1, id2, id3)
        self._zoom_mode = event.key

    def _switch_on_zoom_mode(self, event):
        self._zoom_mode = event.key
        self._mouse_move(event)

    def _switch_off_zoom_mode(self, event):
        self._zoom_mode = None
        self._mouse_move(event)

    def _mouse_move(self, event):
        """Callback for mouse moves in zoom-to-rectangle mode."""
        if self._xypress:
            x, y = (event.x, event.y)
            lastx, lasty, a, ind, view = self._xypress[0]
            (x1, y1), (x2, y2) = np.clip([[lastx, lasty], [x, y]], a.bbox.min, a.bbox.max)
            if self._zoom_mode == 'x':
                y1, y2 = a.bbox.intervaly
            elif self._zoom_mode == 'y':
                x1, x2 = a.bbox.intervalx
            self.toolmanager.trigger_tool('rubberband', self, data=(x1, y1, x2, y2))

    def _release(self, event):
        """Callback for mouse button releases in zoom-to-rectangle mode."""
        for zoom_id in self._ids_zoom:
            self.figure.canvas.mpl_disconnect(zoom_id)
        self._ids_zoom = []
        if not self._xypress:
            self._cancel_action()
            return
        done_ax = []
        for cur_xypress in self._xypress:
            x, y = (event.x, event.y)
            lastx, lasty, a, _ind, view = cur_xypress
            if abs(x - lastx) < 5 or abs(y - lasty) < 5:
                self._cancel_action()
                return
            twinx = any((a.get_shared_x_axes().joined(a, a1) for a1 in done_ax))
            twiny = any((a.get_shared_y_axes().joined(a, a1) for a1 in done_ax))
            done_ax.append(a)
            if self._button_pressed == 1:
                direction = 'in'
            elif self._button_pressed == 3:
                direction = 'out'
            else:
                continue
            a._set_view_from_bbox((lastx, lasty, x, y), direction, self._zoom_mode, twinx, twiny)
        self._zoom_mode = None
        self.toolmanager.get_tool(_views_positions).push_current()
        self._cancel_action()