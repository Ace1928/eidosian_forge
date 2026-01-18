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
def _dataframe_set_geometry(self, col, drop=False, inplace=False, crs=None):
    if inplace:
        raise ValueError("Can't do inplace setting when converting from DataFrame to GeoDataFrame")
    gf = GeoDataFrame(self)
    return gf.set_geometry(col, drop=drop, inplace=False, crs=crs)