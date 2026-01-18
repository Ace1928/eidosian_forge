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
class EckertIII(_Eckert):
    """
    An Eckert III projection.

    This projection is pseudocylindrical, but not equal-area. Parallels are
    equally-spaced straight lines, while meridians are elliptical arcs up to
    semicircles on the edges. Its equal-area pair is :class:`EckertIV`.

    """
    _proj_name = 'eck3'