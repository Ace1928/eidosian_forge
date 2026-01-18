import itertools
import operator
import warnings
import matplotlib
import matplotlib.artist
import matplotlib.collections as mcollections
import matplotlib.text
import matplotlib.ticker as mticker
import matplotlib.transforms as mtrans
import numpy as np
import shapely.geometry as sgeom
import cartopy
from cartopy.crs import PlateCarree, Projection, _RectangularProjection
from cartopy.mpl.ticker import (
def _find_midpoints(self, lim, ticks):
    if len(ticks) > 1:
        cent = np.diff(ticks).mean() / 2
    else:
        cent = np.nan
    if isinstance(self.axes.projection, _POLAR_PROJS):
        lq = 90
        uq = 90
    else:
        lq = 25
        uq = 75
    midpoints = (self._round(np.percentile(lim, lq), cent), self._round(np.percentile(lim, uq), cent))
    return midpoints