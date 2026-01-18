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
class EckertV(_Eckert):
    """
    An Eckert V projection.

    This projection is pseudocylindrical, but not equal-area. Parallels are
    equally-spaced straight lines, while meridians are sinusoidal arcs. Its
    equal-area pair is :class:`EckertVI`.

    """
    _proj_name = 'eck5'