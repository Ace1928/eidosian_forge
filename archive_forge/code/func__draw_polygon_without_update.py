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
def _draw_polygon_without_update(self):
    """Redraw the polygon based on new vertex positions, no update()."""
    xs, ys = zip(*self._xys) if self._xys else ([], [])
    self._selection_artist.set_data(xs, ys)
    self._update_box()
    if self._selection_completed or (len(self._xys) > 3 and self._xys[-1] == self._xys[0]):
        self._polygon_handles.set_data(xs[:-1], ys[:-1])
    else:
        self._polygon_handles.set_data(xs, ys)