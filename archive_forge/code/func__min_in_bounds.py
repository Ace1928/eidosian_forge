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
def _min_in_bounds(self, min):
    """Ensure the new min value is between valmin and self.val[1]."""
    if min <= self.valmin:
        if not self.closedmin:
            return self.val[0]
        min = self.valmin
    if min > self.val[1]:
        min = self.val[1]
    return self._stepped_value(min)