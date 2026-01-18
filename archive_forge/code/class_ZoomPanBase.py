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
class ZoomPanBase(ToolToggleBase):
    """Base class for `ToolZoom` and `ToolPan`."""

    def __init__(self, *args):
        super().__init__(*args)
        self._button_pressed = None
        self._xypress = None
        self._idPress = None
        self._idRelease = None
        self._idScroll = None
        self.base_scale = 2.0
        self.scrollthresh = 0.5
        self.lastscroll = time.time() - self.scrollthresh

    def enable(self, event=None):
        """Connect press/release events and lock the canvas."""
        self.figure.canvas.widgetlock(self)
        self._idPress = self.figure.canvas.mpl_connect('button_press_event', self._press)
        self._idRelease = self.figure.canvas.mpl_connect('button_release_event', self._release)
        self._idScroll = self.figure.canvas.mpl_connect('scroll_event', self.scroll_zoom)

    def disable(self, event=None):
        """Release the canvas and disconnect press/release events."""
        self._cancel_action()
        self.figure.canvas.widgetlock.release(self)
        self.figure.canvas.mpl_disconnect(self._idPress)
        self.figure.canvas.mpl_disconnect(self._idRelease)
        self.figure.canvas.mpl_disconnect(self._idScroll)

    def trigger(self, sender, event, data=None):
        self.toolmanager.get_tool(_views_positions).add_figure(self.figure)
        super().trigger(sender, event, data)
        new_navigate_mode = self.name.upper() if self.toggled else None
        for ax in self.figure.axes:
            ax.set_navigate_mode(new_navigate_mode)

    def scroll_zoom(self, event):
        if event.inaxes is None:
            return
        if event.button == 'up':
            scl = self.base_scale
        elif event.button == 'down':
            scl = 1 / self.base_scale
        else:
            scl = 1
        ax = event.inaxes
        ax._set_view_from_bbox([event.x, event.y, scl])
        if time.time() - self.lastscroll < self.scrollthresh:
            self.toolmanager.get_tool(_views_positions).back()
        self.figure.canvas.draw_idle()
        self.lastscroll = time.time()
        self.toolmanager.get_tool(_views_positions).push_current()