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
def _convert_units(self):
    """Convert bounds of the rectangle."""
    x0 = self.convert_xunits(self._x0)
    y0 = self.convert_yunits(self._y0)
    x1 = self.convert_xunits(self._x0 + self._width)
    y1 = self.convert_yunits(self._y0 + self._height)
    return (x0, y0, x1, y1)