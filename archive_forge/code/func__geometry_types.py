import os
from packaging.version import Version
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype
import pyproj
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry
from geopandas import GeoDataFrame, GeoSeries
from urllib.parse import urlparse as parse_url
from urllib.parse import uses_netloc, uses_params, uses_relative
import urllib.request
def _geometry_types(df):
    """
    Determine the geometry types in the GeoDataFrame for the schema.
    """
    geom_types_2D = df[~df.geometry.has_z].geometry.geom_type.unique()
    geom_types_2D = [gtype for gtype in geom_types_2D if gtype is not None]
    geom_types_3D = df[df.geometry.has_z].geometry.geom_type.unique()
    geom_types_3D = ['3D ' + gtype for gtype in geom_types_3D if gtype is not None]
    geom_types = geom_types_3D + geom_types_2D
    if len(geom_types) == 0:
        return 'Unknown'
    if len(geom_types) == 1:
        geom_types = geom_types[0]
    return geom_types