import numbers
import operator
import warnings
import inspect
from functools import lru_cache
import numpy as np
import pandas as pd
from pandas.api.extensions import (
import shapely
import shapely.affinity
import shapely.geometry
from shapely.geometry.base import BaseGeometry
import shapely.ops
import shapely.wkt
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from . import _compat as compat
from . import _vectorized as vectorized
from .sindex import _get_sindex_class
def _geom_to_shapely(geom):
    """
    Convert internal representation (PyGEOS or Shapely) to external Shapely object.
    """
    if compat.USE_SHAPELY_20:
        return geom
    elif not compat.USE_PYGEOS:
        return geom
    else:
        return vectorized._pygeos_to_shapely(geom)