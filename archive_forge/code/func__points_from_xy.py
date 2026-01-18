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
def _points_from_xy(x, y, z=None):
    if not len(x) == len(y):
        raise ValueError('x and y arrays must be equal length.')
    if z is not None:
        if not len(z) == len(x):
            raise ValueError('z array must be same length as x and y.')
        geom = [shapely.geometry.Point(i, j, k) for i, j, k in zip(x, y, z)]
    else:
        geom = [shapely.geometry.Point(i, j) for i, j in zip(x, y)]
    return geom