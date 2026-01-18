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
class Orthographic(Projection):
    _handles_ellipses = False

    def __init__(self, central_longitude=0.0, central_latitude=0.0, globe=None):
        proj4_params = [('proj', 'ortho'), ('lon_0', central_longitude), ('lat_0', central_latitude)]
        super().__init__(proj4_params, globe=globe)
        a = float(self.ellipsoid.semi_major_metre or WGS84_SEMIMAJOR_AXIS)
        coords = _ellipse_boundary(a * 0.99999, a * 0.99999, n=61)
        self._boundary = sgeom.polygon.LinearRing(coords.T)
        mins = np.min(coords, axis=1)
        maxs = np.max(coords, axis=1)
        self._x_limits = (mins[0], maxs[0])
        self._y_limits = (mins[1], maxs[1])
        self.threshold = np.diff(self._x_limits)[0] * 0.02

    @property
    def boundary(self):
        return self._boundary

    @property
    def x_limits(self):
        return self._x_limits

    @property
    def y_limits(self):
        return self._y_limits