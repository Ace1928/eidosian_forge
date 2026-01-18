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
class _RectangularProjection(Projection, metaclass=ABCMeta):
    """
    The abstract superclass of projections with a rectangular domain which
    is symmetric about the origin.

    """
    _wrappable = True

    def __init__(self, proj4_params, half_width, half_height, globe=None):
        self._half_width = half_width
        self._half_height = half_height
        super().__init__(proj4_params, globe=globe)

    @property
    def boundary(self):
        w, h = (self._half_width, self._half_height)
        return sgeom.LinearRing([(-w, -h), (-w, h), (w, h), (w, -h), (-w, -h)])

    @property
    def x_limits(self):
        return (-self._half_width, self._half_width)

    @property
    def y_limits(self):
        return (-self._half_height, self._half_height)