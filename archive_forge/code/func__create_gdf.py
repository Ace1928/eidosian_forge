from contextlib import contextmanager
import glob
import os
import pathlib
import pandas as pd
import pytest
from geopandas.testing import assert_geodataframe_equal
from geopandas import _compat as compat
import geopandas
from shapely.geometry import Point
def _create_gdf():
    return geopandas.GeoDataFrame({'a': [0.1, 0.2, 0.3], 'geometry': [Point(1, 1), Point(2, 2), Point(3, 3)]}, crs='EPSG:4326')