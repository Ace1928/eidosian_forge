import inspect
import numpy as np
import matplotlib as mpl
from matplotlib import (
from . import art3d, proj3d
def _get_axis_line_edge_points(self, minmax, maxmin, position=None):
    """Get the edge points for the black bolded axis line."""
    mb = [minmax, maxmin]
    mb_rev = mb[::-1]
    mm = [[mb, mb_rev, mb_rev], [mb_rev, mb_rev, mb], [mb, mb, mb]]
    mm = mm[self.axes._vertical_axis][self._axinfo['i']]
    juggled = self._axinfo['juggled']
    edge_point_0 = mm[0].copy()
    if position == 'lower' and mm[1][juggled[-1]] < mm[0][juggled[-1]] or (position == 'upper' and mm[1][juggled[-1]] > mm[0][juggled[-1]]):
        edge_point_0[juggled[-1]] = mm[1][juggled[-1]]
    else:
        edge_point_0[juggled[0]] = mm[1][juggled[0]]
    edge_point_1 = edge_point_0.copy()
    edge_point_1[juggled[1]] = mm[1][juggled[1]]
    return (edge_point_0, edge_point_1)