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
def _to_geo(self, **kwargs):
    """
        Returns a python feature collection (i.e. the geointerface)
        representation of the GeoDataFrame.

        """
    geo = {'type': 'FeatureCollection', 'features': list(self.iterfeatures(**kwargs))}
    if kwargs.get('show_bbox', False):
        geo['bbox'] = tuple(self.total_bounds)
    return geo