import warnings
import weakref
import matplotlib.artist
import matplotlib.collections
import matplotlib.path as mpath
import numpy as np
import cartopy.feature as cfeature
from cartopy.mpl import _MPL_38
import cartopy.mpl.patch as cpatch
class _GeomKey:
    """
    Provide id() based equality and hashing for geometries.

    Instances of this class must be treated as immutable for the caching
    to operate correctly.

    A workaround for Shapely polygons no longer being hashable as of 1.5.13.

    """

    def __init__(self, geom):
        self._id = id(geom)

    def __eq__(self, other):
        return self._id == other._id

    def __hash__(self):
        return hash(self._id)