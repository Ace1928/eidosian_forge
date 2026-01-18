import inspect
import numpy as np
import matplotlib as mpl
from matplotlib import (
from . import art3d, proj3d
def active_pane(self, renderer):
    mins, maxs, centers, deltas, tc, highs = self._get_coord_info(renderer)
    info = self._axinfo
    index = info['i']
    if not highs[index]:
        loc = mins[index]
        plane = self._PLANES[2 * index]
    else:
        loc = maxs[index]
        plane = self._PLANES[2 * index + 1]
    xys = np.array([tc[p] for p in plane])
    return (xys, loc)