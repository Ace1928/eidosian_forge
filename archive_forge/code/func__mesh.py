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
def _mesh(self):
    """
        Return the coordinate arrays for the colorbar pcolormesh/patches.

        These are scaled between vmin and vmax, and already handle colorbar
        orientation.
        """
    y, _ = self._proportional_y()
    if isinstance(self.norm, (colors.BoundaryNorm, colors.NoNorm)) or self.boundaries is not None:
        y = y * (self.vmax - self.vmin) + self.vmin
    else:
        with self.norm.callbacks.blocked(), cbook._setattr_cm(self.norm, vmin=self.vmin, vmax=self.vmax):
            y = self.norm.inverse(y)
    self._y = y
    X, Y = np.meshgrid([0.0, 1.0], y)
    if self.orientation == 'vertical':
        return (X, Y)
    else:
        return (Y, X)