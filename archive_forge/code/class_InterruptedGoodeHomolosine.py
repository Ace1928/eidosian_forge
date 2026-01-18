from abc import ABCMeta
from collections import OrderedDict
from functools import lru_cache
import io
import json
import math
import warnings
import numpy as np
from pyproj import Transformer
from pyproj.exceptions import ProjError
import shapely.geometry as sgeom
from shapely.prepared import prep
import cartopy.trace
class InterruptedGoodeHomolosine(Projection):
    """
    Composite equal-area projection emphasizing either land or
    ocean features.

    Original Reference:
        Goode, J. P., 1925: The Homolosine Projection: A new device for
        portraying the Earth's surface entire. Annals of the
        Association of American Geographers, 15:3, 119-125,
        DOI: 10.1080/00045602509356949

    A central_longitude value of -160 is recommended for the oceanic view.

    """
    _wrappable = True

    def __init__(self, central_longitude=0, globe=None, emphasis='land'):
        """
        Parameters
        ----------
        central_longitude : float, optional
            The central longitude, by default 0
        globe : :class:`cartopy.crs.Globe`, optional
            If omitted, a default Globe object is created, by default None
        emphasis : str, optional
            Options 'land' and 'ocean' are available, by default 'land'
        """
        if emphasis == 'land':
            proj4_params = [('proj', 'igh'), ('lon_0', central_longitude)]
            super().__init__(proj4_params, globe=globe)
        elif emphasis == 'ocean':
            proj4_params = [('proj', 'igh_o'), ('lon_0', central_longitude)]
            super().__init__(proj4_params, globe=globe)
        else:
            msg = "`emphasis` needs to be either 'land' or 'ocean'"
            raise ValueError(msg)
        minlon, maxlon = self._determine_longitude_bounds(central_longitude)
        epsilon = 1e-10
        n = 31
        if emphasis == 'land':
            top_interrupted_lons = (-40.0,)
            bottom_interrupted_lons = (80.0, -20.0, -100.0)
        elif emphasis == 'ocean':
            top_interrupted_lons = (-90.0, 60.0)
            bottom_interrupted_lons = (90.0, -60.0)
        lons = np.empty((2 + 2 * len(top_interrupted_lons + bottom_interrupted_lons)) * n + 1)
        lats = np.empty((2 + 2 * len(top_interrupted_lons + bottom_interrupted_lons)) * n + 1)
        end = 0
        lons[end:end + n] = minlon
        lats[end:end + n] = np.linspace(-90, 90, n)
        end += n
        for lon in top_interrupted_lons:
            lons[end:end + n] = lon - epsilon + central_longitude
            lats[end:end + n] = np.linspace(90, 0, n)
            end += n
            lons[end:end + n] = lon + epsilon + central_longitude
            lats[end:end + n] = np.linspace(0, 90, n)
            end += n
        lons[end:end + n] = maxlon
        lats[end:end + n] = np.linspace(90, -90, n)
        end += n
        for lon in bottom_interrupted_lons:
            lons[end:end + n] = lon + epsilon + central_longitude
            lats[end:end + n] = np.linspace(-90, 0, n)
            end += n
            lons[end:end + n] = lon - epsilon + central_longitude
            lats[end:end + n] = np.linspace(0, -90, n)
            end += n
        lons[-1] = minlon
        lats[-1] = -90
        points = self.transform_points(self.as_geodetic(), lons, lats)
        self._boundary = sgeom.LinearRing(points)
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
        self._x_limits = (mins[0], maxs[0])
        self._y_limits = (mins[1], maxs[1])
        self.threshold = 20000.0

    @property
    def boundary(self):
        return self._boundary

    @property
    def x_limits(self):
        return self._x_limits

    @property
    def y_limits(self):
        return self._y_limits