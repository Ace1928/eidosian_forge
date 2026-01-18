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
class TestFrameSindex:

    def setup_method(self):
        data = {'A': range(5), 'B': range(-5, 0), 'geom': [Point(x, y) for x, y in zip(range(5), range(5))]}
        self.df = GeoDataFrame(data, geometry='geom')

    def test_sindex(self):
        self.df.crs = 'epsg:4326'
        assert self.df.sindex.size == 5
        hits = list(self.df.sindex.intersection((2.5, 2.5, 4, 4)))
        assert len(hits) == 2
        assert hits[0] == 3

    def test_lazy_build(self):
        assert self.df.geometry.values._sindex is None
        assert self.df.sindex.size == 5
        assert self.df.geometry.values._sindex is not None

    def test_sindex_rebuild_on_set_geometry(self):
        assert self.df.sindex is not None
        original_index = self.df.sindex
        self.df.set_geometry([Point(x, y) for x, y in zip(range(5, 10), range(5, 10))], inplace=True)
        assert self.df.sindex is not original_index

    def test_rebuild_on_row_slice(self):
        original_index = self.df.sindex
        sliced = self.df.iloc[:1]
        assert sliced.sindex is not original_index
        original_index = self.df.sindex
        sliced = self.df.iloc[:]
        assert sliced.sindex is original_index
        sliced = self.df.iloc[::-1]
        assert sliced.sindex is not original_index

    def test_rebuild_on_single_col_selection(self):
        """Selecting a single column should not rebuild the spatial index."""
        original_index = self.df.sindex
        geometry_col = self.df['geom']
        assert geometry_col.sindex is original_index
        geometry_col = self.df.geometry
        assert geometry_col.sindex is original_index

    def test_rebuild_on_multiple_col_selection(self):
        """Selecting a subset of columns preserves the index."""
        original_index = self.df.sindex
        subset1 = self.df[['geom', 'A']]
        if compat.PANDAS_GE_20 and (not pd.options.mode.copy_on_write):
            assert subset1.sindex is not original_index
        else:
            assert subset1.sindex is original_index
        subset2 = self.df[['A', 'geom']]
        if compat.PANDAS_GE_20 and (not pd.options.mode.copy_on_write):
            assert subset2.sindex is not original_index
        else:
            assert subset2.sindex is original_index

    def test_rebuild_on_update_inplace(self):
        gdf = self.df.copy()
        old_sindex = gdf.sindex
        gdf.sort_values('A', ascending=False, inplace=True)
        assert not gdf.has_sindex
        new_sindex = gdf.sindex
        assert new_sindex is not old_sindex
        assert gdf.index.tolist() == [4, 3, 2, 1, 0]

    def test_update_inplace_no_rebuild(self):
        gdf = self.df.copy()
        old_sindex = gdf.sindex
        gdf.rename(columns={'A': 'AA'}, inplace=True)
        assert gdf.has_sindex
        new_sindex = gdf.sindex
        assert old_sindex is new_sindex