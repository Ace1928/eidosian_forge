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
def _to_file_fiona(df, filename, driver, schema, crs, mode, **kwargs):
    if schema is None:
        schema = infer_schema(df)
    if crs:
        crs = pyproj.CRS.from_user_input(crs)
    else:
        crs = df.crs
    with fiona_env():
        crs_wkt = None
        try:
            gdal_version = Version(fiona.env.get_gdal_release_name().strip('e'))
        except (AttributeError, ValueError):
            gdal_version = Version('2.0.0')
        if gdal_version >= Version('3.0.0') and crs:
            crs_wkt = crs.to_wkt()
        elif crs:
            crs_wkt = crs.to_wkt('WKT1_GDAL')
        with fiona.open(filename, mode=mode, driver=driver, crs_wkt=crs_wkt, schema=schema, **kwargs) as colxn:
            colxn.writerecords(df.iterfeatures())