from contextlib import ExitStack
import functools
import math
from numbers import Integral
import numpy as np
from numpy import ma
import matplotlib as mpl
from matplotlib import _api, _docstring
from matplotlib.backend_bases import MouseButton
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.text import Text
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll
import matplotlib.font_manager as font_manager
import matplotlib.cbook as cbook
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
def _find_nearest_contour(self, xy, indices=None):
    """
        Find the point in the unfilled contour plot that is closest (in screen
        space) to point *xy*.

        Parameters
        ----------
        xy : tuple[float, float]
            The reference point (in screen space).
        indices : list of int or None, default: None
            Indices of contour levels to consider.  If None (the default), all levels
            are considered.

        Returns
        -------
        idx_level_min : int
            The index of the contour level closest to *xy*.
        idx_vtx_min : int
            The index of the `.Path` segment closest to *xy* (at that level).
        proj : (float, float)
            The point in the contour plot closest to *xy*.
        """
    if self.filled:
        raise ValueError('Method does not support filled contours')
    if indices is None:
        indices = range(len(self._paths))
    d2min = np.inf
    idx_level_min = idx_vtx_min = proj_min = None
    for idx_level in indices:
        path = self._paths[idx_level]
        idx_vtx_start = 0
        for subpath in path._iter_connected_components():
            if not len(subpath.vertices):
                continue
            lc = self.get_transform().transform(subpath.vertices)
            d2, proj, leg = _find_closest_point_on_path(lc, xy)
            if d2 < d2min:
                d2min = d2
                idx_level_min = idx_level
                idx_vtx_min = leg[1] + idx_vtx_start
                proj_min = proj
            idx_vtx_start += len(subpath)
    return (idx_level_min, idx_vtx_min, proj_min)