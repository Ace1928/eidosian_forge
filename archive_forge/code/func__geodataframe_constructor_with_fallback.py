import json
import warnings
import numpy as np
import pandas as pd
import shapely.errors
from pandas import DataFrame, Series
from pandas.core.accessor import CachedAccessor
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from pyproj import CRS
from geopandas.array import GeometryArray, GeometryDtype, from_shapely, to_wkb, to_wkt
from geopandas.base import GeoPandasBase, is_geometry_type
from geopandas.geoseries import GeoSeries
import geopandas.io
from geopandas.explore import _explore
from . import _compat as compat
from ._decorator import doc
def _geodataframe_constructor_with_fallback(*args, **kwargs):
    """
    A flexible constructor for GeoDataFrame._constructor, which falls back
    to returning a DataFrame (if a certain operation does not preserve the
    geometry column)
    """
    df = GeoDataFrame(*args, **kwargs)
    geometry_cols_mask = df.dtypes == 'geometry'
    if len(geometry_cols_mask) == 0 or geometry_cols_mask.sum() == 0:
        df = pd.DataFrame(df)
    return df