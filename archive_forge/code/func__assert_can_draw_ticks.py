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
def _assert_can_draw_ticks(self):
    """
        Check to see if ticks can be drawn. Either returns True or raises
        an exception.

        """
    if not isinstance(self.crs, PlateCarree):
        raise TypeError(f'Cannot label {self.crs.__class__.__name__} gridlines. Only PlateCarree gridlines are currently supported.')
    return True