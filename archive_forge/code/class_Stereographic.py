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
class Stereographic(Projection):
    _wrappable = True

    def __init__(self, central_latitude=0.0, central_longitude=0.0, false_easting=0.0, false_northing=0.0, true_scale_latitude=None, scale_factor=None, globe=None):
        proj4_params = [('proj', 'stere'), ('lat_0', central_latitude), ('lon_0', central_longitude), ('x_0', false_easting), ('y_0', false_northing)]
        if true_scale_latitude is not None:
            if central_latitude not in (-90.0, 90.0):
                warnings.warn('"true_scale_latitude" parameter is only used for polar stereographic projections. Consider the use of "scale_factor" instead.', stacklevel=2)
            proj4_params.append(('lat_ts', true_scale_latitude))
        if scale_factor is not None:
            if true_scale_latitude is not None:
                raise ValueError('It does not make sense to provide both "scale_factor" and "true_scale_latitude". Ignoring "scale_factor".')
            else:
                proj4_params.append(('k_0', scale_factor))
        super().__init__(proj4_params, globe=globe)
        a = float(self.ellipsoid.semi_major_metre or WGS84_SEMIMAJOR_AXIS)
        b = float(self.ellipsoid.semi_minor_metre or WGS84_SEMIMINOR_AXIS)
        x_axis_offset = 50000000.0 / WGS84_SEMIMAJOR_AXIS
        y_axis_offset = 50000000.0 / WGS84_SEMIMINOR_AXIS
        self._x_limits = (-a * x_axis_offset + false_easting, a * x_axis_offset + false_easting)
        self._y_limits = (-b * y_axis_offset + false_northing, b * y_axis_offset + false_northing)
        coords = _ellipse_boundary(self._x_limits[1], self._y_limits[1], false_easting, false_northing, 91)
        self._boundary = sgeom.LinearRing(coords.T)
        self.threshold = np.diff(self._x_limits)[0] * 0.001

    @property
    def boundary(self):
        return self._boundary

    @property
    def x_limits(self):
        return self._x_limits

    @property
    def y_limits(self):
        return self._y_limits