import warnings
import numpy as np
import pandas as pd
import shapely
import shapely.geometry
import shapely.geos
import shapely.ops
import shapely.validation
import shapely.wkb
import shapely.wkt
from shapely.geometry.base import BaseGeometry
from . import _compat as compat
def _unary_geo(op, left, *args, **kwargs):
    """Unary operation that returns new geometries"""
    data = np.empty(len(left), dtype=object)
    with compat.ignore_shapely2_warnings():
        data[:] = [getattr(geom, op, None) for geom in left]
    return data