import warnings
import weakref
import matplotlib.artist
import matplotlib.collections
import matplotlib.path as mpath
import numpy as np
import cartopy.feature as cfeature
from cartopy.mpl import _MPL_38
import cartopy.mpl.patch as cpatch

        Draw the geometries of the feature that intersect with the extent of
        the :class:`cartopy.mpl.geoaxes.GeoAxes` instance to which this
        object has been added.

        