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