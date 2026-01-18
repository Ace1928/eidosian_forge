import math
from typing import Sequence
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point, Polygon, GeometryCollection
import geopandas
import geopandas._compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, sjoin, sjoin_nearest
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest
class TestSpatialJoinNYBB:

    def setup_method(self):
        nybb_filename = geopandas.datasets.get_path('nybb')
        self.polydf = read_file(nybb_filename)
        self.crs = self.polydf.crs
        N = 20
        b = [int(x) for x in self.polydf.total_bounds]
        self.pointdf = GeoDataFrame([{'geometry': Point(x, y), 'pointattr1': x + y, 'pointattr2': x - y} for x, y in zip(range(b[0], b[2], int((b[2] - b[0]) / N)), range(b[1], b[3], int((b[3] - b[1]) / N)))], crs=self.crs)

    def test_geometry_name(self):
        polydf_original_geom_name = self.polydf.geometry.name
        self.polydf = self.polydf.rename(columns={'geometry': 'new_geom'}).set_geometry('new_geom')
        assert polydf_original_geom_name != self.polydf.geometry.name
        res = sjoin(self.polydf, self.pointdf, how='left')
        assert self.polydf.geometry.name == res.geometry.name

    def test_sjoin_left(self):
        df = sjoin(self.pointdf, self.polydf, how='left')
        assert df.shape == (21, 8)
        for i, row in df.iterrows():
            assert row.geometry.geom_type == 'Point'
        assert 'pointattr1' in df.columns
        assert 'BoroCode' in df.columns

    def test_sjoin_right(self):
        df = sjoin(self.pointdf, self.polydf, how='right')
        df2 = sjoin(self.polydf, self.pointdf, how='left')
        assert df.shape == (12, 8)
        assert df.shape == df2.shape
        for i, row in df.iterrows():
            assert row.geometry.geom_type == 'MultiPolygon'
        for i, row in df2.iterrows():
            assert row.geometry.geom_type == 'MultiPolygon'

    def test_sjoin_inner(self):
        df = sjoin(self.pointdf, self.polydf, how='inner')
        assert df.shape == (11, 8)

    def test_sjoin_predicate(self):
        df = sjoin(self.pointdf, self.polydf, how='left', predicate='within')
        assert df.shape == (21, 8)
        assert df.loc[1]['BoroName'] == 'Staten Island'
        df = sjoin(self.pointdf, self.polydf, how='left', predicate='contains')
        assert df.shape == (21, 8)
        assert np.isnan(df.loc[1]['Shape_Area'])

    def test_sjoin_bad_predicate(self):
        with pytest.raises(ValueError):
            sjoin(self.pointdf, self.polydf, how='left', predicate='spandex')

    def test_sjoin_duplicate_column_name(self):
        pointdf2 = self.pointdf.rename(columns={'pointattr1': 'Shape_Area'})
        df = sjoin(pointdf2, self.polydf, how='left')
        assert 'Shape_Area_left' in df.columns
        assert 'Shape_Area_right' in df.columns

    @pytest.mark.parametrize('how', ['left', 'right', 'inner'])
    def test_sjoin_named_index(self, how):
        pointdf2 = self.pointdf.copy()
        pointdf2.index.name = 'pointid'
        polydf = self.polydf.copy()
        polydf.index.name = 'polyid'
        res = sjoin(pointdf2, polydf, how=how)
        assert pointdf2.index.name == 'pointid'
        assert polydf.index.name == 'polyid'
        if how == 'right':
            assert res.index.name == 'polyid'
        else:
            assert res.index.name == 'pointid'

    def test_sjoin_values(self):
        self.polydf.index = [1, 3, 4, 5, 6]
        df = sjoin(self.pointdf, self.polydf, how='left')
        assert df.shape == (21, 8)
        df = sjoin(self.polydf, self.pointdf, how='left')
        assert df.shape == (12, 8)

    @pytest.mark.xfail
    def test_no_overlapping_geometry(self):
        df_inner = sjoin(self.pointdf.iloc[17:], self.polydf, how='inner')
        df_left = sjoin(self.pointdf.iloc[17:], self.polydf, how='left')
        df_right = sjoin(self.pointdf.iloc[17:], self.polydf, how='right')
        expected_inner_df = pd.concat([self.pointdf.iloc[:0], pd.Series(name='index_right', dtype='int64'), self.polydf.drop('geometry', axis=1).iloc[:0]], axis=1)
        expected_inner = GeoDataFrame(expected_inner_df)
        expected_right_df = pd.concat([self.pointdf.drop('geometry', axis=1).iloc[:0], pd.concat([pd.Series(name='index_left', dtype='int64'), pd.Series(name='index_right', dtype='int64')], axis=1), self.polydf], axis=1)
        expected_right = GeoDataFrame(expected_right_df).set_index('index_right')
        expected_left_df = pd.concat([self.pointdf.iloc[17:], pd.Series(name='index_right', dtype='int64'), self.polydf.iloc[:0].drop('geometry', axis=1)], axis=1)
        expected_left = GeoDataFrame(expected_left_df)
        assert expected_inner.equals(df_inner)
        assert expected_right.equals(df_right)
        assert expected_left.equals(df_left)

    @pytest.mark.skip('Not implemented')
    def test_sjoin_outer(self):
        df = sjoin(self.pointdf, self.polydf, how='outer')
        assert df.shape == (21, 8)

    def test_sjoin_empty_geometries(self):
        empty = GeoDataFrame(geometry=[GeometryCollection()] * 3)
        df = sjoin(pd.concat([self.pointdf, empty]), self.polydf, how='left')
        assert df.shape == (24, 8)
        df2 = sjoin(self.pointdf, pd.concat([self.polydf, empty]), how='left')
        assert df2.shape == (21, 8)

    @pytest.mark.parametrize('predicate', ['intersects', 'within', 'contains'])
    def test_sjoin_no_valid_geoms(self, predicate):
        """Tests a completely empty GeoDataFrame."""
        empty = GeoDataFrame(geometry=[], crs=self.pointdf.crs)
        assert sjoin(self.pointdf, empty, how='inner', predicate=predicate).empty
        assert sjoin(self.pointdf, empty, how='right', predicate=predicate).empty
        assert sjoin(empty, self.pointdf, how='inner', predicate=predicate).empty
        assert sjoin(empty, self.pointdf, how='left', predicate=predicate).empty

    def test_empty_sjoin_return_duplicated_columns(self):
        nybb = geopandas.read_file(geopandas.datasets.get_path('nybb'))
        nybb2 = nybb.copy()
        nybb2.geometry = nybb2.translate(200000)
        result = geopandas.sjoin(nybb, nybb2)
        assert 'BoroCode_right' in result.columns
        assert 'BoroCode_left' in result.columns