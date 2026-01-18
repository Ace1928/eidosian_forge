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
class Angle3(_Base):
    """
        Creates a simple quadratic Bézier curve between two points. The middle
        control point is placed at the intersecting point of two lines which
        cross the start and end point, and have a slope of *angleA* and
        *angleB*, respectively.
        """

    def __init__(self, angleA=90, angleB=0):
        """
            Parameters
            ----------
            angleA : float
              Starting angle of the path.

            angleB : float
              Ending angle of the path.
            """
        self.angleA = angleA
        self.angleB = angleB

    def connect(self, posA, posB):
        x1, y1 = posA
        x2, y2 = posB
        cosA = math.cos(math.radians(self.angleA))
        sinA = math.sin(math.radians(self.angleA))
        cosB = math.cos(math.radians(self.angleB))
        sinB = math.sin(math.radians(self.angleB))
        cx, cy = get_intersection(x1, y1, cosA, sinA, x2, y2, cosB, sinB)
        vertices = [(x1, y1), (cx, cy), (x2, y2)]
        codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
        return Path(vertices, codes)