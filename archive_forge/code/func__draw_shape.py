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
def _draw_shape(self, extents):
    x0, x1, y0, y1 = extents
    xmin, xmax = sorted([x0, x1])
    ymin, ymax = sorted([y0, y1])
    center = [x0 + (x1 - x0) / 2.0, y0 + (y1 - y0) / 2.0]
    a = (xmax - xmin) / 2.0
    b = (ymax - ymin) / 2.0
    self._selection_artist.center = center
    self._selection_artist.width = 2 * a
    self._selection_artist.height = 2 * b
    self._selection_artist.angle = self.rotation