from warnings import warn
import warnings
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from shapely.geometry import box, MultiPoint
from shapely.geometry.base import BaseGeometry
from . import _compat as compat
from .array import GeometryArray, GeometryDtype, points_from_xy
def _delegate_binary_method(op, this, other, align, *args, **kwargs):
    this = this.geometry
    if isinstance(other, GeoPandasBase):
        if align and (not this.index.equals(other.index)):
            warn('The indices of the two GeoSeries are different.', stacklevel=4)
            this, other = this.align(other.geometry)
        else:
            other = other.geometry
        a_this = GeometryArray(this.values)
        other = GeometryArray(other.values)
    elif isinstance(other, BaseGeometry):
        a_this = GeometryArray(this.values)
    else:
        raise TypeError(type(this), type(other))
    data = getattr(a_this, op)(other, *args, **kwargs)
    return (data, this.index)