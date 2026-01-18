import pandas as pd
import pyproj
import pytest
from shapely.geometry import Point
import numpy as np
from geopandas import GeoDataFrame, GeoSeries
import geopandas
def _check_metadata_gdf(gdf, geo_name='geometry', crs=crs_wgs):
    assert gdf._geometry_column_name == geo_name
    assert gdf.geometry.name == geo_name
    assert gdf.crs == crs