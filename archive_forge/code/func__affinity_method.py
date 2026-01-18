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
def _affinity_method(op, left, *args, **kwargs):
    left = to_shapely(left)
    out = []
    for geom in left:
        if geom is None or geom.is_empty:
            res = geom
        else:
            res = getattr(shapely.affinity, op)(geom, *args, **kwargs)
        out.append(res)
    data = np.empty(len(left), dtype=object)
    with compat.ignore_shapely2_warnings():
        data[:] = out
    return from_shapely(data)