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
class LambertCylindrical(_RectangularProjection):

    def __init__(self, central_longitude=0.0, globe=None):
        globe = globe or Globe(semimajor_axis=WGS84_SEMIMAJOR_AXIS)
        proj4_params = [('proj', 'cea'), ('lon_0', central_longitude), ('to_meter', math.radians(1) * (globe.semimajor_axis or WGS84_SEMIMAJOR_AXIS))]
        super().__init__(proj4_params, 180, math.degrees(1), globe=globe)