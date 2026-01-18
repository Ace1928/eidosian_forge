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
def _binary_op_float(op, left, right, *args, **kwargs):
    """Binary operation on np.array[geoms] that returns a ndarray"""
    if isinstance(right, BaseGeometry):
        data = [getattr(s, op)(right, *args, **kwargs) if not (s is None or s.is_empty or right.is_empty) else np.nan for s in left]
        return np.array(data, dtype=float)
    elif isinstance(right, np.ndarray):
        if len(left) != len(right):
            msg = 'Lengths of inputs do not match. Left: {0}, Right: {1}'.format(len(left), len(right))
            raise ValueError(msg)
        data = [getattr(this_elem, op)(other_elem, *args, **kwargs) if not (this_elem is None or this_elem.is_empty) | (other_elem is None or other_elem.is_empty) else np.nan for this_elem, other_elem in zip(left, right)]
        return np.array(data, dtype=float)
    else:
        raise TypeError('Type not known: {0} vs {1}'.format(type(left), type(right)))