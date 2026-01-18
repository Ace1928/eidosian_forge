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
def _shapely_to_geom(geom):
    """
    Convert external Shapely object to internal representation (PyGEOS or Shapely).
    """
    if compat.USE_SHAPELY_20:
        return geom
    elif not compat.USE_PYGEOS:
        return geom
    else:
        return vectorized._shapely_to_pygeos(geom)