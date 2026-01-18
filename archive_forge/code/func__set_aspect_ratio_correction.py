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
def _set_aspect_ratio_correction(self):
    aspect_ratio = self.ax._get_aspect_ratio()
    self._selection_artist._aspect_ratio_correction = aspect_ratio
    if self._use_data_coordinates:
        self._aspect_ratio_correction = 1
    else:
        self._aspect_ratio_correction = aspect_ratio