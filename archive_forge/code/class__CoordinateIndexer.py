from warnings import warn
import warnings
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from shapely.geometry import box, MultiPoint
from shapely.geometry.base import BaseGeometry
from . import _compat as compat
from .array import GeometryArray, GeometryDtype, points_from_xy
class _CoordinateIndexer(object):

    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, key):
        obj = self.obj
        xs, ys = key
        if type(xs) is not slice:
            xs = slice(xs, xs)
        if type(ys) is not slice:
            ys = slice(ys, ys)
        if xs.step is not None or ys.step is not None:
            warn('Ignoring step - full interval is used.', stacklevel=2)
        if xs.start is None or xs.stop is None or ys.start is None or (ys.stop is None):
            xmin, ymin, xmax, ymax = obj.total_bounds
        bbox = box(xs.start if xs.start is not None else xmin, ys.start if ys.start is not None else ymin, xs.stop if xs.stop is not None else xmax, ys.stop if ys.stop is not None else ymax)
        idx = obj.intersects(bbox)
        return obj[idx]