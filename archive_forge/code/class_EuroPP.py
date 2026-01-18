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
class EuroPP(UTM):
    """
    UTM Zone 32 projection for EuroPP domain.

    Ellipsoid is International 1924, Datum is ED50.

    """

    def __init__(self):
        globe = Globe(ellipse='intl')
        super().__init__(32, globe=globe)

    @property
    def x_limits(self):
        return (-1400000.0, 2000000.0)

    @property
    def y_limits(self):
        return (4000000.0, 7900000.0)