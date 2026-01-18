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
def _value_in_bounds(self, vals):
    """Clip min, max values to the bounds."""
    return (self._min_in_bounds(vals[0]), self._max_in_bounds(vals[1]))