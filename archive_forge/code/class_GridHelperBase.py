from the axis as some gridlines can never pass any axis.
import numpy as np
import matplotlib as mpl
from matplotlib import _api
import matplotlib.axes as maxes
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import mpl_axes
from .axisline_style import AxislineStyle  # noqa
from .axis_artist import AxisArtist, GridlinesCollection
class GridHelperBase:

    def __init__(self):
        self._old_limits = None
        super().__init__()

    def update_lim(self, axes):
        x1, x2 = axes.get_xlim()
        y1, y2 = axes.get_ylim()
        if self._old_limits != (x1, x2, y1, y2):
            self._update_grid(x1, y1, x2, y2)
            self._old_limits = (x1, x2, y1, y2)

    def _update_grid(self, x1, y1, x2, y2):
        """Cache relevant computations when the axes limits have changed."""

    def get_gridlines(self, which, axis):
        """
        Return list of grid lines as a list of paths (list of points).

        Parameters
        ----------
        which : {"both", "major", "minor"}
        axis : {"both", "x", "y"}
        """
        return []