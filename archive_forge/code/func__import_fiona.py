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
def _import_fiona():
    global fiona
    global fiona_env
    global fiona_import_error
    global FIONA_GE_19
    if fiona is None:
        try:
            import fiona
            try:
                from fiona import Env as fiona_env
            except ImportError:
                try:
                    from fiona import drivers as fiona_env
                except ImportError:
                    fiona_env = None
            FIONA_GE_19 = Version(Version(fiona.__version__).base_version) >= Version('1.9.0')
        except ImportError as err:
            fiona = False
            fiona_import_error = str(err)