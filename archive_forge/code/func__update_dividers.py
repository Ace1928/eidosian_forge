import logging
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, collections, cm, colors, contour, ticker
import matplotlib.artist as martist
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.spines as mspines
import matplotlib.transforms as mtransforms
from matplotlib import _docstring
def _update_dividers(self):
    if not self.drawedges:
        self.dividers.set_segments([])
        return
    if self.orientation == 'vertical':
        lims = self.ax.get_ylim()
        bounds = (lims[0] < self._y) & (self._y < lims[1])
    else:
        lims = self.ax.get_xlim()
        bounds = (lims[0] < self._y) & (self._y < lims[1])
    y = self._y[bounds]
    if self._extend_lower():
        y = np.insert(y, 0, lims[0])
    if self._extend_upper():
        y = np.append(y, lims[1])
    X, Y = np.meshgrid([0, 1], y)
    if self.orientation == 'vertical':
        segments = np.dstack([X, Y])
    else:
        segments = np.dstack([Y, X])
    self.dividers.set_segments(segments)