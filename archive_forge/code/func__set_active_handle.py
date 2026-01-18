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