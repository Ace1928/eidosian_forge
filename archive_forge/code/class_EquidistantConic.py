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
class EquidistantConic(Projection):
    """
    An Equidistant Conic projection.

    This projection is conic and equidistant, and the scale is true along all
    meridians and along one or two specified standard parallels.
    """

    def __init__(self, central_longitude=0.0, central_latitude=0.0, false_easting=0.0, false_northing=0.0, standard_parallels=(20.0, 50.0), globe=None):
        """
        Parameters
        ----------
        central_longitude: optional
            The central longitude. Defaults to 0.
        central_latitude: optional
            The true latitude of the planar origin in degrees. Defaults to 0.
        false_easting: optional
            X offset from planar origin in metres. Defaults to 0.
        false_northing: optional
            Y offset from planar origin in metres. Defaults to 0.
        standard_parallels: optional
            The one or two latitudes of correct scale. Defaults to (20, 50).
        globe: optional
            A :class:`cartopy.crs.Globe`. If omitted, a default globe is
            created.

        """
        proj4_params = [('proj', 'eqdc'), ('lon_0', central_longitude), ('lat_0', central_latitude), ('x_0', false_easting), ('y_0', false_northing)]
        if standard_parallels is not None:
            try:
                proj4_params.append(('lat_1', standard_parallels[0]))
                try:
                    proj4_params.append(('lat_2', standard_parallels[1]))
                except IndexError:
                    pass
            except TypeError:
                proj4_params.append(('lat_1', standard_parallels))
        super().__init__(proj4_params, globe=globe)
        n = 103
        lons = np.empty(2 * n + 1)
        lats = np.empty(2 * n + 1)
        minlon, maxlon = self._determine_longitude_bounds(central_longitude)
        tmp = np.linspace(minlon, maxlon, n)
        lons[:n] = tmp
        lats[:n] = 90
        lons[n:-1] = tmp[::-1]
        lats[n:-1] = -90
        lons[-1] = lons[0]
        lats[-1] = lats[0]
        points = self.transform_points(self.as_geodetic(), lons, lats)
        self._boundary = sgeom.LinearRing(points)
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
        self._x_limits = (mins[0], maxs[0])
        self._y_limits = (mins[1], maxs[1])
        self.threshold = 100000.0

    @property
    def boundary(self):
        return self._boundary

    @property
    def x_limits(self):
        return self._x_limits

    @property
    def y_limits(self):
        return self._y_limits