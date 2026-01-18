import itertools
import time
import numpy as np
import pytest
import shapely.geometry as sgeom
import cartopy.crs as ccrs
class TestSymmetry:

    @pytest.mark.xfail
    def test_curve(self):
        projection = ccrs.PlateCarree()
        coords = [(-0.08, 51.53), (132.0, 43.17)]
        line_string = sgeom.LineString(coords)
        multi_line_string = projection.project_geometry(line_string)
        line_string = sgeom.LineString(coords[::-1])
        multi_line_string2 = projection.project_geometry(line_string)
        assert len(multi_line_string.geoms) == 1
        assert len(multi_line_string2.geoms) == 1
        coords = multi_line_string.geoms[0].coords
        coords2 = multi_line_string2.geoms[0].coords
        np.testing.assert_allclose(coords, coords2[::-1], err_msg='Asymmetric curve generation')