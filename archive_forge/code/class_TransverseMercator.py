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
class TransverseMercator(Projection):
    """
    A Transverse Mercator projection.

    """
    _wrappable = True

    def __init__(self, central_longitude=0.0, central_latitude=0.0, false_easting=0.0, false_northing=0.0, scale_factor=1.0, globe=None, approx=False):
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

        globe: optional
            An instance of :class:`cartopy.crs.Globe`. If omitted, a default
            globe is created.

        approx: optional
            Whether to use Proj's approximate projection (True), or the new
            Extended Transverse Mercator code (False). Defaults to True, but
            will change to False in the next release.

        """
        proj4_params = [('proj', 'tmerc'), ('lon_0', central_longitude), ('lat_0', central_latitude), ('k', scale_factor), ('x_0', false_easting), ('y_0', false_northing), ('units', 'm')]
        if approx:
            proj4_params += [('approx', None)]
        super().__init__(proj4_params, globe=globe)
        self.threshold = 10000.0

    @property
    def boundary(self):
        x0, x1 = self.x_limits
        y0, y1 = self.y_limits
        return sgeom.LinearRing([(x0, y0), (x0, y1), (x1, y1), (x1, y0), (x0, y0)])

    @property
    def x_limits(self):
        return (-20000000.0, 20000000.0)

    @property
    def y_limits(self):
        return (-10000000.0, 10000000.0)