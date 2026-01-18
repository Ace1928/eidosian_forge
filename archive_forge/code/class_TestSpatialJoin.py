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
class TestSpatialJoin:

    @pytest.mark.parametrize('how, lsuffix, rsuffix, expected_cols', [('left', 'left', 'right', {'col_left', 'col_right', 'index_right'}), ('inner', 'left', 'right', {'col_left', 'col_right', 'index_right'}), ('right', 'left', 'right', {'col_left', 'col_right', 'index_left'}), ('left', 'lft', 'rgt', {'col_lft', 'col_rgt', 'index_rgt'}), ('inner', 'lft', 'rgt', {'col_lft', 'col_rgt', 'index_rgt'}), ('right', 'lft', 'rgt', {'col_lft', 'col_rgt', 'index_lft'})])
    def test_suffixes(self, how: str, lsuffix: str, rsuffix: str, expected_cols):
        left = GeoDataFrame({'col': [1], 'geometry': [Point(0, 0)]})
        right = GeoDataFrame({'col': [1], 'geometry': [Point(0, 0)]})
        joined = sjoin(left, right, how=how, lsuffix=lsuffix, rsuffix=rsuffix)
        assert set(joined.columns) == expected_cols | {'geometry'}

    @pytest.mark.parametrize('dfs', ['default-index', 'string-index'], indirect=True)
    def test_crs_mismatch(self, dfs):
        index, df1, df2, expected = dfs
        df1.crs = 'epsg:4326'
        with pytest.warns(UserWarning, match='CRS mismatch between the CRS'):
            sjoin(df1, df2)

    @pytest.mark.parametrize('dfs', ['default-index'], indirect=True)
    @pytest.mark.parametrize('op', ['intersects', 'contains', 'within'])
    def test_deprecated_op_param(self, dfs, op):
        _, df1, df2, _ = dfs
        with pytest.warns(FutureWarning, match='`op` parameter is deprecated'):
            sjoin(df1, df2, op=op)

    @pytest.mark.parametrize('dfs', ['default-index'], indirect=True)
    @pytest.mark.parametrize('op', ['intersects', 'contains', 'within'])
    @pytest.mark.parametrize('predicate', ['contains', 'within'])
    def test_deprecated_op_param_nondefault_predicate(self, dfs, op, predicate):
        _, df1, df2, _ = dfs
        match = 'use the `predicate` parameter instead'
        if op != predicate:
            warntype = UserWarning
            match = '`predicate` will be overridden by the value of `op`' + '(.|\\s)*' + match
        else:
            warntype = FutureWarning
        with pytest.warns(warntype, match=match):
            sjoin(df1, df2, predicate=predicate, op=op)

    @pytest.mark.parametrize('dfs', ['default-index'], indirect=True)
    def test_unknown_kwargs(self, dfs):
        _, df1, df2, _ = dfs
        with pytest.raises(TypeError, match="sjoin\\(\\) got an unexpected keyword argument 'extra_param'"):
            sjoin(df1, df2, extra_param='test')

    @pytest.mark.filterwarnings('ignore:The `op` parameter:FutureWarning')
    @pytest.mark.parametrize('dfs', ['default-index', 'string-index', 'named-index', 'multi-index', 'named-multi-index'], indirect=True)
    @pytest.mark.parametrize('predicate', ['intersects', 'contains', 'within'])
    @pytest.mark.parametrize('predicate_kw', ['predicate', 'op'])
    def test_inner(self, predicate, predicate_kw, dfs):
        index, df1, df2, expected = dfs
        res = sjoin(df1, df2, how='inner', **{predicate_kw: predicate})
        exp = expected[predicate].dropna().copy()
        exp = exp.drop('geometry_y', axis=1).rename(columns={'geometry_x': 'geometry'})
        exp[['df1', 'df2']] = exp[['df1', 'df2']].astype('int64')
        if index == 'default-index':
            exp[['index_left', 'index_right']] = exp[['index_left', 'index_right']].astype('int64')
        if index == 'named-index':
            exp[['df1_ix', 'df2_ix']] = exp[['df1_ix', 'df2_ix']].astype('int64')
            exp = exp.set_index('df1_ix').rename(columns={'df2_ix': 'index_right'})
        if index in ['default-index', 'string-index']:
            exp = exp.set_index('index_left')
            exp.index.name = None
        if index == 'multi-index':
            exp = exp.set_index(['level_0_x', 'level_1_x']).rename(columns={'level_0_y': 'index_right0', 'level_1_y': 'index_right1'})
            exp.index.names = df1.index.names
        if index == 'named-multi-index':
            exp = exp.set_index(['df1_ix1', 'df1_ix2']).rename(columns={'df2_ix1': 'index_right0', 'df2_ix2': 'index_right1'})
            exp.index.names = df1.index.names
        assert_frame_equal(res, exp)

    @pytest.mark.parametrize('dfs', ['default-index', 'string-index', 'named-index', 'multi-index', 'named-multi-index'], indirect=True)
    @pytest.mark.parametrize('predicate', ['intersects', 'contains', 'within'])
    def test_left(self, predicate, dfs):
        index, df1, df2, expected = dfs
        res = sjoin(df1, df2, how='left', predicate=predicate)
        if index in ['default-index', 'string-index']:
            exp = expected[predicate].dropna(subset=['index_left']).copy()
        elif index == 'named-index':
            exp = expected[predicate].dropna(subset=['df1_ix']).copy()
        elif index == 'multi-index':
            exp = expected[predicate].dropna(subset=['level_0_x']).copy()
        elif index == 'named-multi-index':
            exp = expected[predicate].dropna(subset=['df1_ix1']).copy()
        exp = exp.drop('geometry_y', axis=1).rename(columns={'geometry_x': 'geometry'})
        exp['df1'] = exp['df1'].astype('int64')
        if index == 'default-index':
            exp['index_left'] = exp['index_left'].astype('int64')
            res['index_right'] = res['index_right'].astype(float)
        elif index == 'named-index':
            exp[['df1_ix']] = exp[['df1_ix']].astype('int64')
            exp = exp.set_index('df1_ix').rename(columns={'df2_ix': 'index_right'})
        if index in ['default-index', 'string-index']:
            exp = exp.set_index('index_left')
            exp.index.name = None
        if index == 'multi-index':
            exp = exp.set_index(['level_0_x', 'level_1_x']).rename(columns={'level_0_y': 'index_right0', 'level_1_y': 'index_right1'})
            exp.index.names = df1.index.names
        if index == 'named-multi-index':
            exp = exp.set_index(['df1_ix1', 'df1_ix2']).rename(columns={'df2_ix1': 'index_right0', 'df2_ix2': 'index_right1'})
            exp.index.names = df1.index.names
        assert_frame_equal(res, exp)

    def test_empty_join(self):
        polygons = geopandas.GeoDataFrame({'col2': [1, 2], 'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])]})
        not_in = geopandas.GeoDataFrame({'col1': [1], 'geometry': [Point(-0.5, 0.5)]})
        empty = sjoin(not_in, polygons, how='left', predicate='intersects')
        assert empty.index_right.isnull().all()
        empty = sjoin(not_in, polygons, how='right', predicate='intersects')
        assert empty.index_left.isnull().all()
        empty = sjoin(not_in, polygons, how='inner', predicate='intersects')
        assert empty.empty

    @pytest.mark.parametrize('predicate', ['contains', 'contains_properly', 'covered_by', 'covers', 'crosses', 'intersects', 'touches', 'within'])
    @pytest.mark.parametrize('empty', [GeoDataFrame(geometry=[GeometryCollection(), GeometryCollection()]), GeoDataFrame(geometry=GeoSeries())])
    def test_join_with_empty(self, predicate, empty):
        polygons = geopandas.GeoDataFrame({'col2': [1, 2], 'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])]})
        result = sjoin(empty, polygons, how='left', predicate=predicate)
        assert result.index_right.isnull().all()
        result = sjoin(empty, polygons, how='right', predicate=predicate)
        assert result.index_left.isnull().all()
        result = sjoin(empty, polygons, how='inner', predicate=predicate)
        assert result.empty

    @pytest.mark.parametrize('dfs', ['default-index', 'string-index'], indirect=True)
    def test_sjoin_invalid_args(self, dfs):
        index, df1, df2, expected = dfs
        with pytest.raises(ValueError, match="'left_df' should be GeoDataFrame"):
            sjoin(df1.geometry, df2)
        with pytest.raises(ValueError, match="'right_df' should be GeoDataFrame"):
            sjoin(df1, df2.geometry)

    @pytest.mark.parametrize('dfs', ['default-index', 'string-index', 'named-index', 'multi-index', 'named-multi-index'], indirect=True)
    @pytest.mark.parametrize('predicate', ['intersects', 'contains', 'within'])
    def test_right(self, predicate, dfs):
        index, df1, df2, expected = dfs
        res = sjoin(df1, df2, how='right', predicate=predicate)
        if index in ['default-index', 'string-index']:
            exp = expected[predicate].dropna(subset=['index_right']).copy()
        elif index == 'named-index':
            exp = expected[predicate].dropna(subset=['df2_ix']).copy()
        elif index == 'multi-index':
            exp = expected[predicate].dropna(subset=['level_0_y']).copy()
        elif index == 'named-multi-index':
            exp = expected[predicate].dropna(subset=['df2_ix1']).copy()
        exp = exp.drop('geometry_x', axis=1).rename(columns={'geometry_y': 'geometry'})
        exp['df2'] = exp['df2'].astype('int64')
        if index == 'default-index':
            exp['index_right'] = exp['index_right'].astype('int64')
            res['index_left'] = res['index_left'].astype(float)
        elif index == 'named-index':
            exp[['df2_ix']] = exp[['df2_ix']].astype('int64')
            exp = exp.set_index('df2_ix').rename(columns={'df1_ix': 'index_left'})
        if index in ['default-index', 'string-index']:
            exp = exp.set_index('index_right')
            exp = exp.reindex(columns=res.columns)
            exp.index.name = None
        if index == 'multi-index':
            exp = exp.set_index(['level_0_y', 'level_1_y']).rename(columns={'level_0_x': 'index_left0', 'level_1_x': 'index_left1'})
            exp.index.names = df2.index.names
        if index == 'named-multi-index':
            exp = exp.set_index(['df2_ix1', 'df2_ix2']).rename(columns={'df1_ix1': 'index_left0', 'df1_ix2': 'index_left1'})
            exp.index.names = df2.index.names
        if predicate == 'within':
            exp = exp.sort_index()
        assert_frame_equal(res, exp, check_index_type=False)