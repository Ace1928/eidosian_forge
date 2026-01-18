import inspect
import numpy as np
import matplotlib as mpl
from matplotlib import (
from . import art3d, proj3d
def _get_coord_info(self, renderer):
    mins, maxs = np.array([self.axes.get_xbound(), self.axes.get_ybound(), self.axes.get_zbound()]).T
    centers = 0.5 * (maxs + mins)
    deltas = (maxs - mins) / 12
    mins -= 0.25 * deltas
    maxs += 0.25 * deltas
    bounds = (mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2])
    bounds_proj = self.axes._tunit_cube(bounds, self.axes.M)
    means_z0 = np.zeros(3)
    means_z1 = np.zeros(3)
    for i in range(3):
        means_z0[i] = np.mean(bounds_proj[self._PLANES[2 * i], 2])
        means_z1[i] = np.mean(bounds_proj[self._PLANES[2 * i + 1], 2])
    highs = means_z0 < means_z1
    equals = np.abs(means_z0 - means_z1) <= np.finfo(float).eps
    if np.sum(equals) == 2:
        vertical = np.where(~equals)[0][0]
        if vertical == 2:
            highs = np.array([True, True, highs[2]])
        elif vertical == 1:
            highs = np.array([True, highs[1], False])
        elif vertical == 0:
            highs = np.array([highs[0], False, False])
    return (mins, maxs, centers, deltas, bounds_proj, highs)