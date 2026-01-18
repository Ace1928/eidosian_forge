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
class Sinusoidal(Projection):
    """
    A Sinusoidal projection.

    This projection is equal-area.

    """

    def __init__(self, central_longitude=0.0, false_easting=0.0, false_northing=0.0, globe=None):
        """
        Parameters
        ----------
        central_longitude: optional
            The central longitude. Defaults to 0.
        false_easting: optional
            X offset from planar origin in metres. Defaults to 0.
        false_northing: optional
            Y offset from planar origin in metres. Defaults to 0.
        globe: optional
            A :class:`cartopy.crs.Globe`. If omitted, a default globe is
            created.

        """
        proj4_params = [('proj', 'sinu'), ('lon_0', central_longitude), ('x_0', false_easting), ('y_0', false_northing)]
        super().__init__(proj4_params, globe=globe)
        minlon, maxlon = self._determine_longitude_bounds(central_longitude)
        points = []
        n = 91
        lon = np.empty(2 * n + 1)
        lat = np.empty(2 * n + 1)
        lon[:n] = minlon
        lat[:n] = np.linspace(-90, 90, n)
        lon[n:2 * n] = maxlon
        lat[n:2 * n] = np.linspace(90, -90, n)
        lon[-1] = minlon
        lat[-1] = -90
        points = self.transform_points(self.as_geodetic(), lon, lat)
        self._boundary = sgeom.LinearRing(points)
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
        self._x_limits = (mins[0], maxs[0])
        self._y_limits = (mins[1], maxs[1])
        self.threshold = max(np.abs(self.x_limits + self.y_limits)) * 1e-05

    @property
    def boundary(self):
        return self._boundary

    @property
    def x_limits(self):
        return self._x_limits

    @property
    def y_limits(self):
        return self._y_limits