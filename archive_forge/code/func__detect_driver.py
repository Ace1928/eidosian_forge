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
def _detect_driver(path):
    """
    Attempt to auto-detect driver based on the extension
    """
    try:
        path = path.name
    except AttributeError:
        pass
    try:
        return _EXTENSION_TO_DRIVER[Path(path).suffix.lower()]
    except KeyError:
        return 'ESRI Shapefile'