from __future__ import annotations
import json
import typing
from typing import Optional, Any, Callable, Dict
import warnings
import numpy as np
import pandas as pd
from pandas import Series, MultiIndex
from pandas.core.internals import SingleBlockManager
from pyproj import CRS
import shapely
from shapely.geometry.base import BaseGeometry
from shapely.geometry import GeometryCollection
from geopandas.base import GeoPandasBase, _delegate_property
from geopandas.plotting import plot_series
from geopandas.explore import _explore_geoseries
import geopandas
from . import _compat as compat
from ._decorator import doc
from .array import (
from .base import is_geometry_type
def _wrapped_pandas_method(self, mtd, *args, **kwargs):
    """Wrap a generic pandas method to ensure it returns a GeoSeries"""
    val = getattr(super(), mtd)(*args, **kwargs)
    if type(val) == Series:
        val.__class__ = GeoSeries
        val.crs = self.crs
    return val