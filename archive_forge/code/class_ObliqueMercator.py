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
class ObliqueMercator(Projection):
    """
    An Oblique Mercator projection.

    """
    _wrappable = True

    def __init__(self, central_longitude=0.0, central_latitude=0.0, false_easting=0.0, false_northing=0.0, scale_factor=1.0, azimuth=0.0, globe=None):
        """
        Parameters
        ----------
        central_longitude: optional
            The true longitude of the central meridian in degrees.
            Defaults to 0.
        central_latitude: optional
            The true latitude of the planar origin in degrees. Defaults to 0.
        false_easting: optional
            X offset from the planar origin in metres. Defaults to 0.
        false_northing: optional
            Y offset from the planar origin in metres. Defaults to 0.
        scale_factor: optional
            Scale factor at the central meridian. Defaults to 1.
        azimuth: optional
            Azimuth of centerline clockwise from north at the center point of
            the centre line. Defaults to 0.
        globe: optional
            An instance of :class:`cartopy.crs.Globe`. If omitted, a default
            globe is created.

        Notes
        -----
        The 'Rotated Mercator' projection can be achieved using Oblique
        Mercator with `azimuth` ``=90``.

        """
        if np.isclose(azimuth, 90):
            azimuth -= 0.001
        proj4_params = [('proj', 'omerc'), ('lonc', central_longitude), ('lat_0', central_latitude), ('k', scale_factor), ('x_0', false_easting), ('y_0', false_northing), ('alpha', azimuth), ('units', 'm')]
        super().__init__(proj4_params, globe=globe)
        mercator = Mercator(central_longitude=central_longitude, globe=globe, false_easting=false_easting, false_northing=false_northing, scale_factor=scale_factor)
        self._x_limits = mercator.x_limits
        self._y_limits = mercator.y_limits
        self.threshold = mercator.threshold

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