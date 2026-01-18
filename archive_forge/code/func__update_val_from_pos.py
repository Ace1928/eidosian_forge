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
def _update_val_from_pos(self, pos):
    """Update the slider value based on a given position."""
    idx = np.argmin(np.abs(self.val - pos))
    if idx == 0:
        val = self._min_in_bounds(pos)
        self.set_min(val)
    else:
        val = self._max_in_bounds(pos)
        self.set_max(val)
    if self._active_handle:
        if self.orientation == 'vertical':
            self._active_handle.set_ydata([val])
        else:
            self._active_handle.set_xdata([val])