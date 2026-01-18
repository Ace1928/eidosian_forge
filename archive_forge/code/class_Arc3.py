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
@_register_style(_style_list)
class Arc3(_Base):
    """
        Creates a simple quadratic Bézier curve between two
        points. The curve is created so that the middle control point
        (C1) is located at the same distance from the start (C0) and
        end points(C2) and the distance of the C1 to the line
        connecting C0-C2 is *rad* times the distance of C0-C2.
        """

    def __init__(self, rad=0.0):
        """
            Parameters
            ----------
            rad : float
              Curvature of the curve.
            """
        self.rad = rad

    def connect(self, posA, posB):
        x1, y1 = posA
        x2, y2 = posB
        x12, y12 = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        dx, dy = (x2 - x1, y2 - y1)
        f = self.rad
        cx, cy = (x12 + f * dy, y12 - f * dx)
        vertices = [(x1, y1), (cx, cy), (x2, y2)]
        codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
        return Path(vertices, codes)