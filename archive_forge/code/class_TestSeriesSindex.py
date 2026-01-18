from math import sqrt
from shapely.geometry import (
from numpy.testing import assert_array_equal
import geopandas
from geopandas import _compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, datasets
import pytest
import numpy as np
import pandas as pd
@pytest.mark.skip_no_sindex
class TestSeriesSindex:

    def test_has_sindex(self):
        """Test the has_sindex method."""
        t1 = Polygon([(0, 0), (1, 0), (1, 1)])
        t2 = Polygon([(0, 0), (1, 1), (0, 1)])
        d = GeoDataFrame({'geom': [t1, t2]}, geometry='geom')
        assert not d.has_sindex
        d.sindex
        assert d.has_sindex
        d.geometry.values._sindex = None
        assert not d.has_sindex
        d.sindex
        assert d.has_sindex
        s = GeoSeries([t1, t2])
        assert not s.has_sindex
        s.sindex
        assert s.has_sindex
        s.values._sindex = None
        assert not s.has_sindex
        s.sindex
        assert s.has_sindex

    def test_empty_geoseries(self):
        """Tests creating a spatial index from an empty GeoSeries."""
        s = GeoSeries(dtype=object)
        assert not s.sindex
        assert len(s.sindex) == 0

    def test_point(self):
        s = GeoSeries([Point(0, 0)])
        assert s.sindex.size == 1
        hits = s.sindex.intersection((-1, -1, 1, 1))
        assert len(list(hits)) == 1
        hits = s.sindex.intersection((-2, -2, -1, -1))
        assert len(list(hits)) == 0

    def test_empty_point(self):
        """Tests that a single empty Point results in an empty tree."""
        s = GeoSeries([Point()])
        assert not s.sindex
        assert len(s.sindex) == 0

    def test_polygons(self):
        t1 = Polygon([(0, 0), (1, 0), (1, 1)])
        t2 = Polygon([(0, 0), (1, 1), (0, 1)])
        sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        s = GeoSeries([t1, t2, sq])
        assert s.sindex.size == 3

    @pytest.mark.filterwarnings('ignore:The series.append method is deprecated')
    @pytest.mark.skipif(compat.PANDAS_GE_20, reason='append removed in pandas 2.0')
    def test_polygons_append(self):
        t1 = Polygon([(0, 0), (1, 0), (1, 1)])
        t2 = Polygon([(0, 0), (1, 1), (0, 1)])
        sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        s = GeoSeries([t1, t2, sq])
        t = GeoSeries([t1, t2, sq], [3, 4, 5])
        s = s.append(t)
        assert len(s) == 6
        assert s.sindex.size == 6

    def test_lazy_build(self):
        s = GeoSeries([Point(0, 0)])
        assert s.values._sindex is None
        assert s.sindex.size == 1
        assert s.values._sindex is not None

    def test_rebuild_on_item_change(self):
        s = GeoSeries([Point(0, 0)])
        original_index = s.sindex
        s.iloc[0] = Point(0, 0)
        assert s.sindex is not original_index

    def test_rebuild_on_slice(self):
        s = GeoSeries([Point(0, 0), Point(0, 0)])
        original_index = s.sindex
        sliced = s.iloc[:1]
        assert sliced.sindex is not original_index
        sliced = s.iloc[:]
        assert sliced.sindex is original_index
        sliced = s.iloc[::-1]
        assert sliced.sindex is not original_index