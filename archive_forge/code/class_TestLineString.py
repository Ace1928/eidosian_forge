import itertools
import time
import numpy as np
import pytest
import shapely.geometry as sgeom
import cartopy.crs as ccrs
class TestLineString:

    def test_out_of_bounds(self):
        projection = ccrs.TransverseMercator(central_longitude=0, approx=True)
        start_points = [(86, 0), (130, 0)]
        end_points = [(88, 0), (120, 0)]
        for start, end in itertools.product(start_points, end_points):
            line_string = sgeom.LineString([start, end])
            multi_line_string = projection.project_geometry(line_string)
            if start[0] == 130 and end[0] == 120:
                expected = 0
            else:
                expected = 1
            assert len(multi_line_string.geoms) == expected, f'Unexpected line when working from {start} to {end}'

    def test_simple_fragment_count(self):
        projection = ccrs.PlateCarree()
        tests = [([(150, 0), (-150, 0)], 2), ([(10, 0), (90, 0), (180, 0), (-90, 0), (-10, 0)], 2), ([(-10, 0), (10, 0)], 1), ([(-45, 0), (45, 30)], 1)]
        for coords, pieces in tests:
            line_string = sgeom.LineString(coords)
            multi_line_string = projection.project_geometry(line_string)
            assert len(multi_line_string.geoms) == pieces

    def test_split(self):
        projection = ccrs.Robinson(170.5)
        line_string = sgeom.LineString([(-10, 30), (10, 60)])
        multi_line_string = projection.project_geometry(line_string)
        assert len(multi_line_string.geoms) == 2

    def test_out_of_domain_efficiency(self):
        line_string = sgeom.LineString([(0, -90), (2, -90)])
        tgt_proj = ccrs.NorthPolarStereo()
        src_proj = ccrs.PlateCarree()
        cutoff_time = time.time() + 1
        tgt_proj.project_geometry(line_string, src_proj)
        assert time.time() < cutoff_time, 'Projection took too long'