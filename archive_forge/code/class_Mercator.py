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
class Mercator(Projection):
    """
    A Mercator projection.

    """
    _wrappable = True

    def __init__(self, central_longitude=0.0, min_latitude=-80.0, max_latitude=84.0, globe=None, latitude_true_scale=None, false_easting=0.0, false_northing=0.0, scale_factor=None):
        """
        Parameters
        ----------
        central_longitude: optional
            The central longitude. Defaults to 0.
        min_latitude: optional
            The maximum southerly extent of the projection. Defaults
            to -80 degrees.
        max_latitude: optional
            The maximum northerly extent of the projection. Defaults
            to 84 degrees.
        globe: A :class:`cartopy.crs.Globe`, optional
            If omitted, a default globe is created.
        latitude_true_scale: optional
            The latitude where the scale is 1. Defaults to 0 degrees.
        false_easting: optional
            X offset from the planar origin in metres. Defaults to 0.
        false_northing: optional
            Y offset from the planar origin in metres. Defaults to 0.
        scale_factor: optional
            Scale factor at natural origin. Defaults to unused.

        Notes
        -----
        Only one of ``latitude_true_scale`` and ``scale_factor`` should
        be included.
        """
        proj4_params = [('proj', 'merc'), ('lon_0', central_longitude), ('x_0', false_easting), ('y_0', false_northing), ('units', 'm')]
        if latitude_true_scale is not None:
            proj4_params.append(('lat_ts', latitude_true_scale))
        if scale_factor is not None:
            if latitude_true_scale is not None:
                raise ValueError('It does not make sense to provide both "scale_factor" and "latitude_true_scale". ')
            else:
                proj4_params.append(('k_0', scale_factor))
        super().__init__(proj4_params, globe=globe)
        self._x_limits = self._y_limits = None
        minlon, maxlon = self._determine_longitude_bounds(central_longitude)
        limits = self.transform_points(self.as_geodetic(), np.array([minlon, maxlon]), np.array([min_latitude, max_latitude]))
        self._x_limits = tuple(limits[..., 0])
        self._y_limits = tuple(limits[..., 1])
        self.threshold = min(np.diff(self.x_limits)[0] / 720, np.diff(self.y_limits)[0] / 360)

    def __eq__(self, other):
        res = super().__eq__(other)
        if hasattr(other, '_y_limits') and hasattr(other, '_x_limits'):
            res = res and self._y_limits == other._y_limits and (self._x_limits == other._x_limits)
        return res

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.proj4_init, self._x_limits, self._y_limits))

    @property
    def boundary(self):
        x0, x1 = self.x_limits
        y0, y1 = self.y_limits
        return sgeom.LinearRing([(x0, y0), (x0, y1), (x1, y1), (x1, y0), (x0, y0)])

    @property
    def x_limits(self):
        return self._x_limits

    @property
    def y_limits(self):
        return self._y_limits