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
def _update_box(self):
    if self._box is not None:
        bbox = self._get_bbox()
        self._box.extents = [bbox.x0, bbox.x1, bbox.y0, bbox.y1]
        self._old_box_extents = self._box.extents