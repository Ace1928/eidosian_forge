from __future__ import annotations
import copy
from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core.indexes import (
from xarray.core.variable import IndexVariable, Variable
from xarray.tests import assert_array_equal, assert_identical, requires_cftime
from xarray.tests.test_coding_times import _all_cftime_date_types
class TestPandasMultiIndex:

    def test_constructor(self) -> None:
        foo_data = np.array([0, 0, 1], dtype='int64')
        bar_data = np.array([1.1, 1.2, 1.3], dtype='float64')
        pd_idx = pd.MultiIndex.from_arrays([foo_data, bar_data], names=('foo', 'bar'))
        index = PandasMultiIndex(pd_idx, 'x')
        assert index.dim == 'x'
        assert index.index.equals(pd_idx)
        assert index.index.names == ('foo', 'bar')
        assert index.index.name == 'x'
        assert index.level_coords_dtype == {'foo': foo_data.dtype, 'bar': bar_data.dtype}
        with pytest.raises(ValueError, match='.*conflicting multi-index level name.*'):
            PandasMultiIndex(pd_idx, 'foo')
        pd_idx = pd.MultiIndex.from_arrays([foo_data, bar_data])
        index = PandasMultiIndex(pd_idx, 'x')
        assert list(index.index.names) == ['x_level_0', 'x_level_1']

    def test_from_variables(self) -> None:
        v_level1 = xr.Variable('x', [1, 2, 3], attrs={'unit': 'm'}, encoding={'dtype': np.int32})
        v_level2 = xr.Variable('x', ['a', 'b', 'c'], attrs={'unit': 'm'}, encoding={'dtype': 'U'})
        index = PandasMultiIndex.from_variables({'level1': v_level1, 'level2': v_level2}, options={})
        expected_idx = pd.MultiIndex.from_arrays([v_level1.data, v_level2.data])
        assert index.dim == 'x'
        assert index.index.equals(expected_idx)
        assert index.index.name == 'x'
        assert list(index.index.names) == ['level1', 'level2']
        var = xr.Variable(('x', 'y'), [[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError, match='.*only accepts 1-dimensional variables.*'):
            PandasMultiIndex.from_variables({'var': var}, options={})
        v_level3 = xr.Variable('y', [4, 5, 6])
        with pytest.raises(ValueError, match='unmatched dimensions for multi-index variables.*'):
            PandasMultiIndex.from_variables({'level1': v_level1, 'level3': v_level3}, options={})

    def test_concat(self) -> None:
        pd_midx = pd.MultiIndex.from_product([[0, 1, 2], ['a', 'b']], names=('foo', 'bar'))
        level_coords_dtype = {'foo': np.int32, 'bar': '=U1'}
        midx1 = PandasMultiIndex(pd_midx[:2], 'x', level_coords_dtype=level_coords_dtype)
        midx2 = PandasMultiIndex(pd_midx[2:], 'x', level_coords_dtype=level_coords_dtype)
        expected = PandasMultiIndex(pd_midx, 'x', level_coords_dtype=level_coords_dtype)
        actual = PandasMultiIndex.concat([midx1, midx2], 'x')
        assert actual.equals(expected)
        assert actual.level_coords_dtype == expected.level_coords_dtype

    def test_stack(self) -> None:
        prod_vars = {'x': xr.Variable('x', pd.Index(['b', 'a']), attrs={'foo': 'bar'}), 'y': xr.Variable('y', pd.Index([1, 3, 2]))}
        index = PandasMultiIndex.stack(prod_vars, 'z')
        assert index.dim == 'z'
        assert list(index.index.names) == ['x', 'y']
        np.testing.assert_array_equal(index.index.codes, [[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]])
        with pytest.raises(ValueError, match='conflicting dimensions for multi-index product.*'):
            PandasMultiIndex.stack({'x': xr.Variable('x', ['a', 'b']), 'x2': xr.Variable('x', [1, 2])}, 'z')

    def test_stack_non_unique(self) -> None:
        prod_vars = {'x': xr.Variable('x', pd.Index(['b', 'a']), attrs={'foo': 'bar'}), 'y': xr.Variable('y', pd.Index([1, 1, 2]))}
        index = PandasMultiIndex.stack(prod_vars, 'z')
        np.testing.assert_array_equal(index.index.codes, [[0, 0, 0, 1, 1, 1], [0, 0, 1, 0, 0, 1]])
        np.testing.assert_array_equal(index.index.levels[0], ['b', 'a'])
        np.testing.assert_array_equal(index.index.levels[1], [1, 2])

    def test_unstack(self) -> None:
        pd_midx = pd.MultiIndex.from_product([['a', 'b'], [1, 2, 3]], names=['one', 'two'])
        index = PandasMultiIndex(pd_midx, 'x')
        new_indexes, new_pd_idx = index.unstack()
        assert list(new_indexes) == ['one', 'two']
        assert new_indexes['one'].equals(PandasIndex(['a', 'b'], 'one'))
        assert new_indexes['two'].equals(PandasIndex([1, 2, 3], 'two'))
        assert new_pd_idx.equals(pd_midx)

    def test_unstack_requires_unique(self) -> None:
        pd_midx = pd.MultiIndex.from_product([['a', 'a'], [1, 2]], names=['one', 'two'])
        index = PandasMultiIndex(pd_midx, 'x')
        with pytest.raises(ValueError, match='Cannot unstack MultiIndex containing duplicates'):
            index.unstack()

    def test_create_variables(self) -> None:
        foo_data = np.array([0, 0, 1], dtype='int64')
        bar_data = np.array([1.1, 1.2, 1.3], dtype='float64')
        pd_idx = pd.MultiIndex.from_arrays([foo_data, bar_data], names=('foo', 'bar'))
        index_vars = {'x': IndexVariable('x', pd_idx), 'foo': IndexVariable('x', foo_data, attrs={'unit': 'm'}), 'bar': IndexVariable('x', bar_data, encoding={'fill_value': 0})}
        index = PandasMultiIndex(pd_idx, 'x')
        actual = index.create_variables(index_vars)
        for k, expected in index_vars.items():
            assert_identical(actual[k], expected)
            assert actual[k].dtype == expected.dtype
            if k != 'x':
                assert actual[k].dtype == index.level_coords_dtype[k]

    def test_sel(self) -> None:
        index = PandasMultiIndex(pd.MultiIndex.from_product([['a', 'b'], [1, 2]], names=('one', 'two')), 'x')
        actual = index.sel({'x': slice(('a', 1), ('b', 2))})
        expected_dim_indexers = {'x': slice(0, 4)}
        assert actual.dim_indexers == expected_dim_indexers
        with pytest.raises(KeyError, match='not all values found'):
            index.sel({'x': [0]})
        with pytest.raises(KeyError):
            index.sel({'x': 0})
        with pytest.raises(ValueError, match='cannot provide labels for both.*'):
            index.sel({'one': 0, 'x': 'a'})
        with pytest.raises(ValueError, match="multi-index level names \\('three',\\) not found in indexes"):
            index.sel({'x': {'three': 0}})
        with pytest.raises(IndexError):
            index.sel({'x': (slice(None), 1, 'no_level')})

    def test_join(self):
        midx = pd.MultiIndex.from_product([['a', 'aa'], [1, 2]], names=('one', 'two'))
        level_coords_dtype = {'one': '=U2', 'two': 'i'}
        index1 = PandasMultiIndex(midx, 'x', level_coords_dtype=level_coords_dtype)
        index2 = PandasMultiIndex(midx[0:2], 'x', level_coords_dtype=level_coords_dtype)
        actual = index1.join(index2)
        assert actual.equals(index2)
        assert actual.level_coords_dtype == level_coords_dtype
        actual = index1.join(index2, how='outer')
        assert actual.equals(index1)
        assert actual.level_coords_dtype == level_coords_dtype

    def test_rename(self) -> None:
        level_coords_dtype = {'one': '<U1', 'two': np.int32}
        index = PandasMultiIndex(pd.MultiIndex.from_product([['a', 'b'], [1, 2]], names=('one', 'two')), 'x', level_coords_dtype=level_coords_dtype)
        new_index = index.rename({}, {})
        assert new_index is index
        new_index = index.rename({'two': 'three'}, {})
        assert list(new_index.index.names) == ['one', 'three']
        assert new_index.dim == 'x'
        assert new_index.level_coords_dtype == {'one': '<U1', 'three': np.int32}
        new_index = index.rename({}, {'x': 'y'})
        assert list(new_index.index.names) == ['one', 'two']
        assert new_index.dim == 'y'
        assert new_index.level_coords_dtype == level_coords_dtype

    def test_copy(self) -> None:
        level_coords_dtype = {'one': 'U<1', 'two': np.int32}
        expected = PandasMultiIndex(pd.MultiIndex.from_product([['a', 'b'], [1, 2]], names=('one', 'two')), 'x', level_coords_dtype=level_coords_dtype)
        actual = expected.copy()
        assert actual.index.equals(expected.index)
        assert actual.index is not expected.index
        assert actual.dim == expected.dim
        assert actual.level_coords_dtype == expected.level_coords_dtype