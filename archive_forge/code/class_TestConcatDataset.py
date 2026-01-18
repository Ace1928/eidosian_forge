from __future__ import annotations
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable
import numpy as np
import pandas as pd
import pytest
from xarray import DataArray, Dataset, Variable, concat
from xarray.core import dtypes, merge
from xarray.core.coordinates import Coordinates
from xarray.core.indexes import PandasIndex
from xarray.tests import (
from xarray.tests.test_dataset import create_test_data
class TestConcatDataset:

    @pytest.fixture
    def data(self) -> Dataset:
        return create_test_data().drop_dims('dim3')

    def rectify_dim_order(self, data, dataset) -> Dataset:
        return Dataset({k: v.transpose(*data[k].dims) for k, v in dataset.data_vars.items()}, dataset.coords, attrs=dataset.attrs)

    @pytest.mark.parametrize('coords', ['different', 'minimal'])
    @pytest.mark.parametrize('dim', ['dim1', 'dim2'])
    def test_concat_simple(self, data, dim, coords) -> None:
        datasets = [g for _, g in data.groupby(dim, squeeze=False)]
        assert_identical(data, concat(datasets, dim, coords=coords))

    def test_concat_merge_variables_present_in_some_datasets(self, data) -> None:
        ds1 = Dataset(data_vars={'a': ('y', [0.1])}, coords={'x': 0.1})
        ds2 = Dataset(data_vars={'a': ('y', [0.2])}, coords={'z': 0.2})
        actual = concat([ds1, ds2], dim='y', coords='minimal')
        expected = Dataset({'a': ('y', [0.1, 0.2])}, coords={'x': 0.1, 'z': 0.2})
        assert_identical(expected, actual)
        split_data = [data.isel(dim1=slice(3)), data.isel(dim1=slice(3, None))]
        data0, data1 = deepcopy(split_data)
        data1['foo'] = ('bar', np.random.randn(10))
        actual = concat([data0, data1], 'dim1', data_vars='minimal')
        expected = data.copy().assign(foo=data1.foo)
        assert_identical(expected, actual)
        actual = concat([data0, data1], 'dim1')
        foo = np.ones((8, 10), dtype=data1.foo.dtype) * np.nan
        foo[3:] = data1.foo.values[None, ...]
        expected = data.copy().assign(foo=(['dim1', 'bar'], foo))
        assert_identical(expected, actual)

    def test_concat_2(self, data) -> None:
        dim = 'dim2'
        datasets = [g.squeeze(dim) for _, g in data.groupby(dim, squeeze=False)]
        concat_over = [k for k, v in data.coords.items() if dim in v.dims and k != dim]
        actual = concat(datasets, data[dim], coords=concat_over)
        assert_identical(data, self.rectify_dim_order(data, actual))

    @pytest.mark.parametrize('coords', ['different', 'minimal', 'all'])
    @pytest.mark.parametrize('dim', ['dim1', 'dim2'])
    def test_concat_coords_kwarg(self, data, dim, coords) -> None:
        data = data.copy(deep=True)
        data.coords['extra'] = ('dim4', np.arange(3))
        datasets = [g.squeeze() for _, g in data.groupby(dim, squeeze=False)]
        actual = concat(datasets, data[dim], coords=coords)
        if coords == 'all':
            expected = np.array([data['extra'].values for _ in range(data.sizes[dim])])
            assert_array_equal(actual['extra'].values, expected)
        else:
            assert_equal(data['extra'], actual['extra'])

    def test_concat(self, data) -> None:
        split_data = [data.isel(dim1=slice(3)), data.isel(dim1=3), data.isel(dim1=slice(4, None))]
        assert_identical(data, concat(split_data, 'dim1'))

    def test_concat_dim_precedence(self, data) -> None:
        dim = (2 * data['dim1']).rename('dim1')
        datasets = [g for _, g in data.groupby('dim1', squeeze=False)]
        expected = data.copy()
        expected['dim1'] = dim
        assert_identical(expected, concat(datasets, dim))

    def test_concat_data_vars_typing(self) -> None:
        data = Dataset({'foo': ('x', np.random.randn(10))})
        objs: list[Dataset] = [data.isel(x=slice(5)), data.isel(x=slice(5, None))]
        actual = concat(objs, dim='x', data_vars='minimal')
        assert_identical(data, actual)

    def test_concat_data_vars(self) -> None:
        data = Dataset({'foo': ('x', np.random.randn(10))})
        objs: list[Dataset] = [data.isel(x=slice(5)), data.isel(x=slice(5, None))]
        for data_vars in ['minimal', 'different', 'all', [], ['foo']]:
            actual = concat(objs, dim='x', data_vars=data_vars)
            assert_identical(data, actual)

    def test_concat_coords(self):
        data = Dataset({'foo': ('x', np.random.randn(10))})
        expected = data.assign_coords(c=('x', [0] * 5 + [1] * 5))
        objs = [data.isel(x=slice(5)).assign_coords(c=0), data.isel(x=slice(5, None)).assign_coords(c=1)]
        for coords in ['different', 'all', ['c']]:
            actual = concat(objs, dim='x', coords=coords)
            assert_identical(expected, actual)
        for coords in ['minimal', []]:
            with pytest.raises(merge.MergeError, match='conflicting values'):
                concat(objs, dim='x', coords=coords)

    def test_concat_constant_index(self):
        ds1 = Dataset({'foo': 1.5}, {'y': 1})
        ds2 = Dataset({'foo': 2.5}, {'y': 1})
        expected = Dataset({'foo': ('y', [1.5, 2.5]), 'y': [1, 1]})
        for mode in ['different', 'all', ['foo']]:
            actual = concat([ds1, ds2], 'y', data_vars=mode)
            assert_identical(expected, actual)
        with pytest.raises(merge.MergeError, match='conflicting values'):
            concat([ds1, ds2], 'new_dim', data_vars='minimal')

    def test_concat_size0(self) -> None:
        data = create_test_data()
        split_data = [data.isel(dim1=slice(0, 0)), data]
        actual = concat(split_data, 'dim1')
        assert_identical(data, actual)
        actual = concat(split_data[::-1], 'dim1')
        assert_identical(data, actual)

    def test_concat_autoalign(self) -> None:
        ds1 = Dataset({'foo': DataArray([1, 2], coords=[('x', [1, 2])])})
        ds2 = Dataset({'foo': DataArray([1, 2], coords=[('x', [1, 3])])})
        actual = concat([ds1, ds2], 'y')
        expected = Dataset({'foo': DataArray([[1, 2, np.nan], [1, np.nan, 2]], dims=['y', 'x'], coords={'x': [1, 2, 3]})})
        assert_identical(expected, actual)

    def test_concat_errors(self):
        data = create_test_data()
        split_data = [data.isel(dim1=slice(3)), data.isel(dim1=slice(3, None))]
        with pytest.raises(ValueError, match='must supply at least one'):
            concat([], 'dim1')
        with pytest.raises(ValueError, match="Cannot specify both .*='different'"):
            concat([data, data], dim='concat_dim', data_vars='different', compat='override')
        with pytest.raises(ValueError, match='must supply at least one'):
            concat([], 'dim1')
        with pytest.raises(ValueError, match='are not found in the coordinates'):
            concat([data, data], 'new_dim', coords=['not_found'])
        with pytest.raises(ValueError, match='are not found in the data variables'):
            concat([data, data], 'new_dim', data_vars=['not_found'])
        with pytest.raises(ValueError, match='global attributes not'):
            data0 = deepcopy(split_data[0])
            data1 = deepcopy(split_data[1])
            data1.attrs['foo'] = 'bar'
            concat([data0, data1], 'dim1', compat='identical')
        assert_identical(data, concat([data0, data1], 'dim1', compat='equals'))
        with pytest.raises(ValueError, match='compat.* invalid'):
            concat(split_data, 'dim1', compat='foobar')
        with pytest.raises(ValueError, match='unexpected value for'):
            concat([data, data], 'new_dim', coords='foobar')
        with pytest.raises(ValueError, match='coordinate in some datasets but not others'):
            concat([Dataset({'x': 0}), Dataset({'x': [1]})], dim='z')
        with pytest.raises(ValueError, match='coordinate in some datasets but not others'):
            concat([Dataset({'x': 0}), Dataset({}, {'x': 1})], dim='z')

    def test_concat_join_kwarg(self) -> None:
        ds1 = Dataset({'a': (('x', 'y'), [[0]])}, coords={'x': [0], 'y': [0]})
        ds2 = Dataset({'a': (('x', 'y'), [[0]])}, coords={'x': [1], 'y': [0.0001]})
        expected: dict[JoinOptions, Any] = {}
        expected['outer'] = Dataset({'a': (('x', 'y'), [[0, np.nan], [np.nan, 0]])}, {'x': [0, 1], 'y': [0, 0.0001]})
        expected['inner'] = Dataset({'a': (('x', 'y'), [[], []])}, {'x': [0, 1], 'y': []})
        expected['left'] = Dataset({'a': (('x', 'y'), np.array([0, np.nan], ndmin=2).T)}, coords={'x': [0, 1], 'y': [0]})
        expected['right'] = Dataset({'a': (('x', 'y'), np.array([np.nan, 0], ndmin=2).T)}, coords={'x': [0, 1], 'y': [0.0001]})
        expected['override'] = Dataset({'a': (('x', 'y'), np.array([0, 0], ndmin=2).T)}, coords={'x': [0, 1], 'y': [0]})
        with pytest.raises(ValueError, match="cannot align.*exact.*dimensions.*'y'"):
            actual = concat([ds1, ds2], join='exact', dim='x')
        for join in expected:
            actual = concat([ds1, ds2], join=join, dim='x')
            assert_equal(actual, expected[join])
        actual = concat([ds1.drop_vars('x'), ds2.drop_vars('x')], join='override', dim='y')
        expected2 = Dataset({'a': (('x', 'y'), np.array([0, 0], ndmin=2))}, coords={'y': [0, 0.0001]})
        assert_identical(actual, expected2)

    @pytest.mark.parametrize('combine_attrs, var1_attrs, var2_attrs, expected_attrs, expect_exception', [('no_conflicts', {'a': 1, 'b': 2}, {'a': 1, 'c': 3}, {'a': 1, 'b': 2, 'c': 3}, False), ('no_conflicts', {'a': 1, 'b': 2}, {}, {'a': 1, 'b': 2}, False), ('no_conflicts', {}, {'a': 1, 'c': 3}, {'a': 1, 'c': 3}, False), ('no_conflicts', {'a': 1, 'b': 2}, {'a': 4, 'c': 3}, {'a': 1, 'b': 2, 'c': 3}, True), ('drop', {'a': 1, 'b': 2}, {'a': 1, 'c': 3}, {}, False), ('identical', {'a': 1, 'b': 2}, {'a': 1, 'b': 2}, {'a': 1, 'b': 2}, False), ('identical', {'a': 1, 'b': 2}, {'a': 1, 'c': 3}, {'a': 1, 'b': 2}, True), ('override', {'a': 1, 'b': 2}, {'a': 4, 'b': 5, 'c': 3}, {'a': 1, 'b': 2}, False), ('drop_conflicts', {'a': 41, 'b': 42, 'c': 43}, {'b': 2, 'c': 43, 'd': 44}, {'a': 41, 'c': 43, 'd': 44}, False), (lambda attrs, context: {'a': -1, 'b': 0, 'c': 1} if any(attrs) else {}, {'a': 41, 'b': 42, 'c': 43}, {'b': 2, 'c': 43, 'd': 44}, {'a': -1, 'b': 0, 'c': 1}, False)])
    def test_concat_combine_attrs_kwarg(self, combine_attrs, var1_attrs, var2_attrs, expected_attrs, expect_exception):
        ds1 = Dataset({'a': ('x', [0])}, coords={'x': [0]}, attrs=var1_attrs)
        ds2 = Dataset({'a': ('x', [0])}, coords={'x': [1]}, attrs=var2_attrs)
        if expect_exception:
            with pytest.raises(ValueError, match=f"combine_attrs='{combine_attrs}'"):
                concat([ds1, ds2], dim='x', combine_attrs=combine_attrs)
        else:
            actual = concat([ds1, ds2], dim='x', combine_attrs=combine_attrs)
            expected = Dataset({'a': ('x', [0, 0])}, {'x': [0, 1]}, attrs=expected_attrs)
            assert_identical(actual, expected)

    @pytest.mark.parametrize('combine_attrs, attrs1, attrs2, expected_attrs, expect_exception', [('no_conflicts', {'a': 1, 'b': 2}, {'a': 1, 'c': 3}, {'a': 1, 'b': 2, 'c': 3}, False), ('no_conflicts', {'a': 1, 'b': 2}, {}, {'a': 1, 'b': 2}, False), ('no_conflicts', {}, {'a': 1, 'c': 3}, {'a': 1, 'c': 3}, False), ('no_conflicts', {'a': 1, 'b': 2}, {'a': 4, 'c': 3}, {'a': 1, 'b': 2, 'c': 3}, True), ('drop', {'a': 1, 'b': 2}, {'a': 1, 'c': 3}, {}, False), ('identical', {'a': 1, 'b': 2}, {'a': 1, 'b': 2}, {'a': 1, 'b': 2}, False), ('identical', {'a': 1, 'b': 2}, {'a': 1, 'c': 3}, {'a': 1, 'b': 2}, True), ('override', {'a': 1, 'b': 2}, {'a': 4, 'b': 5, 'c': 3}, {'a': 1, 'b': 2}, False), ('drop_conflicts', {'a': 41, 'b': 42, 'c': 43}, {'b': 2, 'c': 43, 'd': 44}, {'a': 41, 'c': 43, 'd': 44}, False), (lambda attrs, context: {'a': -1, 'b': 0, 'c': 1} if any(attrs) else {}, {'a': 41, 'b': 42, 'c': 43}, {'b': 2, 'c': 43, 'd': 44}, {'a': -1, 'b': 0, 'c': 1}, False)])
    def test_concat_combine_attrs_kwarg_variables(self, combine_attrs, attrs1, attrs2, expected_attrs, expect_exception):
        """check that combine_attrs is used on data variables and coords"""
        ds1 = Dataset({'a': ('x', [0], attrs1)}, coords={'x': ('x', [0], attrs1)})
        ds2 = Dataset({'a': ('x', [0], attrs2)}, coords={'x': ('x', [1], attrs2)})
        if expect_exception:
            with pytest.raises(ValueError, match=f"combine_attrs='{combine_attrs}'"):
                concat([ds1, ds2], dim='x', combine_attrs=combine_attrs)
        else:
            actual = concat([ds1, ds2], dim='x', combine_attrs=combine_attrs)
            expected = Dataset({'a': ('x', [0, 0], expected_attrs)}, {'x': ('x', [0, 1], expected_attrs)})
            assert_identical(actual, expected)

    def test_concat_promote_shape(self) -> None:
        objs = [Dataset({}, {'x': 0}), Dataset({'x': [1]})]
        actual = concat(objs, 'x')
        expected = Dataset({'x': [0, 1]})
        assert_identical(actual, expected)
        objs = [Dataset({'x': [0]}), Dataset({}, {'x': 1})]
        actual = concat(objs, 'x')
        assert_identical(actual, expected)
        objs = [Dataset({'x': [2], 'y': 3}), Dataset({'x': [4], 'y': 5})]
        actual = concat(objs, 'x')
        expected = Dataset({'x': [2, 4], 'y': ('x', [3, 5])})
        assert_identical(actual, expected)
        objs = [Dataset({'x': [0]}, {'y': -1}), Dataset({'x': [1]}, {'y': ('x', [-2])})]
        actual = concat(objs, 'x')
        expected = Dataset({'x': [0, 1]}, {'y': ('x', [-1, -2])})
        assert_identical(actual, expected)
        objs = [Dataset({'x': [0]}, {'y': -1}), Dataset({'x': [1, 2]}, {'y': -2})]
        actual = concat(objs, 'x')
        expected = Dataset({'x': [0, 1, 2]}, {'y': ('x', [-1, -2, -2])})
        assert_identical(actual, expected)
        objs = [Dataset({'z': ('x', [-1])}, {'x': [0], 'y': [0]}), Dataset({'z': ('y', [1])}, {'x': [1], 'y': [0]})]
        actual = concat(objs, 'x')
        expected = Dataset({'z': (('x', 'y'), [[-1], [1]])}, {'x': [0, 1], 'y': [0]})
        assert_identical(actual, expected)
        objs = [Dataset({}, {'x': pd.Interval(-1, 0, closed='right')}), Dataset({'x': [pd.Interval(0, 1, closed='right')]})]
        actual = concat(objs, 'x')
        expected = Dataset({'x': [pd.Interval(-1, 0, closed='right'), pd.Interval(0, 1, closed='right')]})
        assert_identical(actual, expected)
        time_data1 = np.array(['2022-01-01', '2022-02-01'], dtype='datetime64[ns]')
        time_data2 = np.array('2022-03-01', dtype='datetime64[ns]')
        time_expected = np.array(['2022-01-01', '2022-02-01', '2022-03-01'], dtype='datetime64[ns]')
        objs = [Dataset({}, {'time': time_data1}), Dataset({}, {'time': time_data2})]
        actual = concat(objs, 'time')
        expected = Dataset({}, {'time': time_expected})
        assert_identical(actual, expected)
        assert isinstance(actual.indexes['time'], pd.DatetimeIndex)

    def test_concat_do_not_promote(self) -> None:
        objs = [Dataset({'y': ('t', [1])}, {'x': 1, 't': [0]}), Dataset({'y': ('t', [2])}, {'x': 1, 't': [0]})]
        expected = Dataset({'y': ('t', [1, 2])}, {'x': 1, 't': [0, 0]})
        actual = concat(objs, 't')
        assert_identical(expected, actual)
        objs = [Dataset({'y': ('t', [1])}, {'x': 1, 't': [0]}), Dataset({'y': ('t', [2])}, {'x': 2, 't': [0]})]
        with pytest.raises(ValueError):
            concat(objs, 't', coords='minimal')

    def test_concat_dim_is_variable(self) -> None:
        objs = [Dataset({'x': 0}), Dataset({'x': 1})]
        coord = Variable('y', [3, 4], attrs={'foo': 'bar'})
        expected = Dataset({'x': ('y', [0, 1]), 'y': coord})
        actual = concat(objs, coord)
        assert_identical(actual, expected)

    def test_concat_dim_is_dataarray(self) -> None:
        objs = [Dataset({'x': 0}), Dataset({'x': 1})]
        coord = DataArray([3, 4], dims='y', attrs={'foo': 'bar'})
        expected = Dataset({'x': ('y', [0, 1]), 'y': coord})
        actual = concat(objs, coord)
        assert_identical(actual, expected)

    def test_concat_multiindex(self) -> None:
        midx = pd.MultiIndex.from_product([[1, 2, 3], ['a', 'b']])
        midx_coords = Coordinates.from_pandas_multiindex(midx, 'x')
        expected = Dataset(coords=midx_coords)
        actual = concat([expected.isel(x=slice(2)), expected.isel(x=slice(2, None))], 'x')
        assert expected.equals(actual)
        assert isinstance(actual.x.to_index(), pd.MultiIndex)

    def test_concat_along_new_dim_multiindex(self) -> None:
        level_names = ['x_level_0', 'x_level_1']
        midx = pd.MultiIndex.from_product([[1, 2, 3], ['a', 'b']], names=level_names)
        midx_coords = Coordinates.from_pandas_multiindex(midx, 'x')
        ds = Dataset(coords=midx_coords)
        concatenated = concat([ds], 'new')
        actual = list(concatenated.xindexes.get_all_coords('x'))
        expected = ['x'] + level_names
        assert actual == expected

    @pytest.mark.parametrize('fill_value', [dtypes.NA, 2, 2.0, {'a': 2, 'b': 1}])
    def test_concat_fill_value(self, fill_value) -> None:
        datasets = [Dataset({'a': ('x', [2, 3]), 'b': ('x', [-2, 1]), 'x': [1, 2]}), Dataset({'a': ('x', [1, 2]), 'b': ('x', [3, -1]), 'x': [0, 1]})]
        if fill_value == dtypes.NA:
            fill_value_a = fill_value_b = np.nan
        elif isinstance(fill_value, dict):
            fill_value_a = fill_value['a']
            fill_value_b = fill_value['b']
        else:
            fill_value_a = fill_value_b = fill_value
        expected = Dataset({'a': (('t', 'x'), [[fill_value_a, 2, 3], [1, 2, fill_value_a]]), 'b': (('t', 'x'), [[fill_value_b, -2, 1], [3, -1, fill_value_b]])}, {'x': [0, 1, 2]})
        actual = concat(datasets, dim='t', fill_value=fill_value)
        assert_identical(actual, expected)

    @pytest.mark.parametrize('dtype', [str, bytes])
    @pytest.mark.parametrize('dim', ['x1', 'x2'])
    def test_concat_str_dtype(self, dtype, dim) -> None:
        data = np.arange(4).reshape([2, 2])
        da1 = Dataset({'data': (['x1', 'x2'], data), 'x1': [0, 1], 'x2': np.array(['a', 'b'], dtype=dtype)})
        da2 = Dataset({'data': (['x1', 'x2'], data), 'x1': np.array([1, 2]), 'x2': np.array(['c', 'd'], dtype=dtype)})
        actual = concat([da1, da2], dim=dim)
        assert np.issubdtype(actual.x2.dtype, dtype)