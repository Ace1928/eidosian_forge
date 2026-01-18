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
def _import_pyogrio():
    global pyogrio
    global pyogrio_import_error
    if pyogrio is None:
        try:
            import pyogrio
        except ImportError as err:
            pyogrio = False
            pyogrio_import_error = str(err)