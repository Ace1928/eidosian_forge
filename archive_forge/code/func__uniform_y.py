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
def _uniform_y(self, N):
    """
        Return colorbar data coordinates for *N* uniformly
        spaced boundaries, plus extension lengths if required.
        """
    automin = automax = 1.0 / (N - 1.0)
    extendlength = self._get_extension_lengths(self.extendfrac, automin, automax, default=0.05)
    y = np.linspace(0, 1, N)
    return (y, extendlength)