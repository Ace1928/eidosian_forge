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
def _get_common_crs(arr_seq):
    arr_seq = [ga for ga in arr_seq if not (ga.isna().all() and ga.crs is None)]
    unique_crs = []
    for arr in arr_seq:
        if arr.crs not in unique_crs:
            unique_crs.append(arr.crs)
    crs_not_none = [crs for crs in unique_crs if crs is not None]
    names = [crs.name for crs in crs_not_none]
    if len(crs_not_none) == 0:
        return None
    if len(crs_not_none) == 1:
        if len(unique_crs) != 1:
            warnings.warn(f"CRS not set for some of the concatenation inputs. Setting output's CRS as {names[0]} (the single non-null crs provided).", stacklevel=2)
        return crs_not_none[0]
    raise ValueError(f'Cannot determine common CRS for concatenation inputs, got {names}. Use `to_crs()` to transform geometries to the same CRS before merging.')