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
class OSNI(TransverseMercator):

    def __init__(self, approx=False):
        globe = Globe(semimajor_axis=6377340.189, semiminor_axis=6356034.447938534)
        super().__init__(central_longitude=-8, central_latitude=53.5, scale_factor=1.000035, false_easting=200000, false_northing=250000, globe=globe, approx=approx)

    @property
    def boundary(self):
        w = self.x_limits[1] - self.x_limits[0]
        h = self.y_limits[1] - self.y_limits[0]
        return sgeom.LinearRing([(0, 0), (0, h), (w, h), (w, 0), (0, 0)])

    @property
    def x_limits(self):
        return (18814.9667, 386062.3293)

    @property
    def y_limits(self):
        return (11764.8481, 464720.9559)