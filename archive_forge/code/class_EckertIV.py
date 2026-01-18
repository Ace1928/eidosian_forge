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
class EckertIV(_Eckert):
    """
    An Eckert IV projection.

    This projection is pseudocylindrical, and equal-area. Parallels are
    unequally-spaced straight lines, while meridians are elliptical arcs up to
    semicircles on the edges. Its non-equal-area pair with equally-spaced
    parallels is :class:`EckertIII`.

    It is commonly used for world maps.

    """
    _proj_name = 'eck4'