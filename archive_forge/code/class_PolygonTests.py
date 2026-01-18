import numpy as np
import pytest
import shapely.geometry as sgeom
import shapely.wkt
import cartopy.crs as ccrs
class PolygonTests:

    def _assert_bounds(self, bounds, x1, y1, x2, y2, delta=1):
        assert abs(bounds[0] - x1) < delta
        assert abs(bounds[1] - y1) < delta
        assert abs(bounds[2] - x2) < delta
        assert abs(bounds[3] - y2) < delta