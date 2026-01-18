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
def _expanddim_logic(df):
    """Shared logic for _constructor_expanddim and _constructor_from_mgr_expanddim."""
    from geopandas import GeoDataFrame
    if (df.dtypes == 'geometry').sum() > 0:
        if df.shape[1] == 1:
            geo_col_name = df.columns[0]
        else:
            geo_col_name = None
        if geo_col_name is None or not is_geometry_type(df[geo_col_name]):
            df = GeoDataFrame(df)
            df._geometry_column_name = None
        else:
            df = df.set_geometry(geo_col_name)
    return df