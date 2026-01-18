import math
import numpy as np
from contextlib import contextmanager
from matplotlib import (
from matplotlib.collections import (
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from . import proj3d
def _maybe_depth_shade_and_sort_colors(self, color_array):
    color_array = _zalpha(color_array, self._vzs) if self._vzs is not None and self._depthshade else color_array
    if len(color_array) > 1:
        color_array = color_array[self._z_markers_idx]
    return mcolors.to_rgba_array(color_array, self._alpha)