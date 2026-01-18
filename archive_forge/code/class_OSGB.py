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
class OSGB(TransverseMercator):

    def __init__(self, approx=False):
        super().__init__(central_longitude=-2, central_latitude=49, scale_factor=0.9996012717, false_easting=400000, false_northing=-100000, globe=Globe(datum='OSGB36', ellipse='airy'), approx=approx)

    @property
    def boundary(self):
        w = self.x_limits[1] - self.x_limits[0]
        h = self.y_limits[1] - self.y_limits[0]
        return sgeom.LinearRing([(0, 0), (0, h), (w, h), (w, 0), (0, 0)])

    @property
    def x_limits(self):
        return (0, 700000.0)

    @property
    def y_limits(self):
        return (0, 1300000.0)