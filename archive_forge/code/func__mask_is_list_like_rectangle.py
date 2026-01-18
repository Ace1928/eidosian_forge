import warnings
import numpy as np
import pandas.api.types
from shapely.geometry import Polygon, MultiPolygon, box
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import _check_crs, _crs_mismatch_warn
def _mask_is_list_like_rectangle(mask):
    return pandas.api.types.is_list_like(mask) and (not isinstance(mask, (GeoDataFrame, GeoSeries, Polygon, MultiPolygon)))