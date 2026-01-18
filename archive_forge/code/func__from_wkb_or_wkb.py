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
@classmethod
def _from_wkb_or_wkb(cls, from_wkb_or_wkt_function: Callable, data, index=None, crs: Optional[Any]=None, **kwargs) -> GeoSeries:
    """Create a GeoSeries from either WKT or WKB values"""
    if isinstance(data, Series):
        if index is not None:
            data = data.reindex(index)
        else:
            index = data.index
        data = data.values
    return cls(from_wkb_or_wkt_function(data, crs=crs), index=index, **kwargs)