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
def _generate_labels(self):
    """
        A generator to yield as many labels as needed, re-using existing ones
        where possible.
        """
    for label in self._all_labels:
        yield label
    while True:
        new_artist = matplotlib.text.Text()
        new_artist.set_figure(self.axes.figure)
        new_artist.axes = self.axes
        new_label = Label(new_artist, None, None, None)
        self._all_labels.append(new_label)
        yield new_label