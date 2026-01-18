import os
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame
from geopandas.testing import assert_geodataframe_equal
import pytest
from .test_file import FIONA_MARK, PYOGRIO_MARK
def _expect_writing(gdf, ogr_driver):
    return _ExpectedErrorBuilder(_composite_key(gdf, ogr_driver))