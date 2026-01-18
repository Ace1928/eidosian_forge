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
def _hover(self, event):
    """Update the canvas cursor if it's over a handle."""
    if self.ignore(event):
        return
    if self._active_handle is not None or not self._selection_completed:
        return
    _, e_dist = self._edge_handles.closest(event.x, event.y)
    self._set_cursor(e_dist <= self.grab_range)