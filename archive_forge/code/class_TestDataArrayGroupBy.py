from __future__ import annotations
import datetime
import operator
import warnings
from unittest import mock
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import xarray as xr
from xarray import DataArray, Dataset, Variable
from xarray.core.groupby import _consolidate_slices
from xarray.core.types import InterpOptions
from xarray.tests import (
class TestDataArrayGroupBy:

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.attrs = {'attr1': 'value1', 'attr2': 2929}
        self.x = np.random.random((10, 20))
        self.v = Variable(['x', 'y'], self.x)
        self.va = Variable(['x', 'y'], self.x, self.attrs)
        self.ds = Dataset({'foo': self.v})
        self.dv = self.ds['foo']
        self.mindex = pd.MultiIndex.from_product([['a', 'b'], [1, 2]], names=('level_1', 'level_2'))
        self.mda = DataArray([0, 1, 2, 3], coords={'x': self.mindex}, dims='x')
        self.da = self.dv.copy()
        self.da.coords['abc'] = ('y', np.array(['a'] * 9 + ['c'] + ['b'] * 10))
        self.da.coords['y'] = 20 + 100 * self.da['y']

    def test_stack_groupby_unsorted_coord(self) -> None:
        data = [[0, 1], [2, 3]]
        data_flat = [0, 1, 2, 3]
        dims = ['x', 'y']
        y_vals = [2, 3]
        arr = xr.DataArray(data, dims=dims, coords={'y': y_vals})
        actual1 = arr.stack(z=dims).groupby('z').first()
        midx1 = pd.MultiIndex.from_product([[0, 1], [2, 3]], names=dims)
        expected1 = xr.DataArray(data_flat, dims=['z'], coords={'z': midx1})
        assert_equal(actual1, expected1)
        arr = xr.DataArray(data, dims=dims, coords={'y': y_vals[::-1]})
        actual2 = arr.stack(z=dims).groupby('z').first()
        midx2 = pd.MultiIndex.from_product([[0, 1], [3, 2]], names=dims)
        expected2 = xr.DataArray(data_flat, dims=['z'], coords={'z': midx2})
        assert_equal(actual2, expected2)

    def test_groupby_iter(self) -> None:
        for (act_x, act_dv), (exp_x, exp_ds) in zip(self.dv.groupby('y', squeeze=False), self.ds.groupby('y', squeeze=False)):
            assert exp_x == act_x
            assert_identical(exp_ds['foo'], act_dv)
        with pytest.warns(UserWarning, match='The `squeeze` kwarg'):
            for (_, exp_dv), (_, act_dv) in zip(self.dv.groupby('x'), self.dv.groupby('x')):
                assert_identical(exp_dv, act_dv)

    def test_groupby_properties(self) -> None:
        grouped = self.da.groupby('abc')
        expected_groups = {'a': range(0, 9), 'c': [9], 'b': range(10, 20)}
        assert expected_groups.keys() == grouped.groups.keys()
        for key in expected_groups:
            expected_group = expected_groups[key]
            actual_group = grouped.groups[key]
            assert not isinstance(expected_group, slice)
            assert not isinstance(actual_group, slice)
            np.testing.assert_array_equal(expected_group, actual_group)
        assert 3 == len(grouped)

    @pytest.mark.parametrize('by, use_da', [('x', False), ('y', False), ('y', True), ('abc', False)])
    @pytest.mark.parametrize('shortcut', [True, False])
    @pytest.mark.parametrize('squeeze', [None, True, False])
    def test_groupby_map_identity(self, by, use_da, shortcut, squeeze, recwarn) -> None:
        expected = self.da
        if use_da:
            by = expected.coords[by]

        def identity(x):
            return x
        grouped = expected.groupby(by, squeeze=squeeze)
        actual = grouped.map(identity, shortcut=shortcut)
        assert_identical(expected, actual)
        if (by.name if use_da else by) != 'abc':
            assert len(recwarn) == (1 if squeeze in [None, True] else 0)

    def test_groupby_sum(self) -> None:
        array = self.da
        grouped = array.groupby('abc')
        expected_sum_all = Dataset({'foo': Variable(['abc'], np.array([self.x[:, :9].sum(), self.x[:, 10:].sum(), self.x[:, 9:10].sum()]).T), 'abc': Variable(['abc'], np.array(['a', 'b', 'c']))})['foo']
        assert_allclose(expected_sum_all, grouped.reduce(np.sum, dim=...))
        assert_allclose(expected_sum_all, grouped.sum(...))
        expected = DataArray([array['y'].values[idx].sum() for idx in [slice(9), slice(10, None), slice(9, 10)]], [['a', 'b', 'c']], ['abc'])
        actual = array['y'].groupby('abc').map(np.sum)
        assert_allclose(expected, actual)
        actual = array['y'].groupby('abc').sum(...)
        assert_allclose(expected, actual)
        expected_sum_axis1 = Dataset({'foo': (['x', 'abc'], np.array([self.x[:, :9].sum(1), self.x[:, 10:].sum(1), self.x[:, 9:10].sum(1)]).T), 'abc': Variable(['abc'], np.array(['a', 'b', 'c']))})['foo']
        assert_allclose(expected_sum_axis1, grouped.reduce(np.sum, 'y'))
        assert_allclose(expected_sum_axis1, grouped.sum('y'))

    @pytest.mark.parametrize('method', ['sum', 'mean', 'median'])
    def test_groupby_reductions(self, method) -> None:
        array = self.da
        grouped = array.groupby('abc')
        reduction = getattr(np, method)
        expected = Dataset({'foo': Variable(['x', 'abc'], np.array([reduction(self.x[:, :9], axis=-1), reduction(self.x[:, 10:], axis=-1), reduction(self.x[:, 9:10], axis=-1)]).T), 'abc': Variable(['abc'], np.array(['a', 'b', 'c']))})['foo']
        with xr.set_options(use_flox=False):
            actual_legacy = getattr(grouped, method)(dim='y')
        with xr.set_options(use_flox=True):
            actual_npg = getattr(grouped, method)(dim='y')
        assert_allclose(expected, actual_legacy)
        assert_allclose(expected, actual_npg)

    def test_groupby_count(self) -> None:
        array = DataArray([0, 0, np.nan, np.nan, 0, 0], coords={'cat': ('x', ['a', 'b', 'b', 'c', 'c', 'c'])}, dims='x')
        actual = array.groupby('cat').count()
        expected = DataArray([1, 1, 2], coords=[('cat', ['a', 'b', 'c'])])
        assert_identical(actual, expected)

    @pytest.mark.parametrize('shortcut', [True, False])
    @pytest.mark.parametrize('keep_attrs', [None, True, False])
    def test_groupby_reduce_keep_attrs(self, shortcut: bool, keep_attrs: bool | None) -> None:
        array = self.da
        array.attrs['foo'] = 'bar'
        actual = array.groupby('abc').reduce(np.mean, keep_attrs=keep_attrs, shortcut=shortcut)
        with xr.set_options(use_flox=False):
            expected = array.groupby('abc').mean(keep_attrs=keep_attrs)
        assert_identical(expected, actual)

    @pytest.mark.parametrize('keep_attrs', [None, True, False])
    def test_groupby_keep_attrs(self, keep_attrs: bool | None) -> None:
        array = self.da
        array.attrs['foo'] = 'bar'
        with xr.set_options(use_flox=False):
            expected = array.groupby('abc').mean(keep_attrs=keep_attrs)
        with xr.set_options(use_flox=True):
            actual = array.groupby('abc').mean(keep_attrs=keep_attrs)
        actual.data = expected.data
        assert_identical(expected, actual)

    def test_groupby_map_center(self) -> None:

        def center(x):
            return x - np.mean(x)
        array = self.da
        grouped = array.groupby('abc')
        expected_ds = array.to_dataset()
        exp_data = np.hstack([center(self.x[:, :9]), center(self.x[:, 9:10]), center(self.x[:, 10:])])
        expected_ds['foo'] = (['x', 'y'], exp_data)
        expected_centered = expected_ds['foo']
        assert_allclose(expected_centered, grouped.map(center))

    def test_groupby_map_ndarray(self) -> None:
        array = self.da
        grouped = array.groupby('abc')
        actual = grouped.map(np.asarray)
        assert_equal(array, actual)

    def test_groupby_map_changes_metadata(self) -> None:

        def change_metadata(x):
            x.coords['x'] = x.coords['x'] * 2
            x.attrs['fruit'] = 'lemon'
            return x
        array = self.da
        grouped = array.groupby('abc')
        actual = grouped.map(change_metadata)
        expected = array.copy()
        expected = change_metadata(expected)
        assert_equal(expected, actual)

    @pytest.mark.parametrize('squeeze', [True, False])
    def test_groupby_math_squeeze(self, squeeze: bool) -> None:
        array = self.da
        grouped = array.groupby('x', squeeze=squeeze)
        expected = array + array.coords['x']
        actual = grouped + array.coords['x']
        assert_identical(expected, actual)
        actual = array.coords['x'] + grouped
        assert_identical(expected, actual)
        ds = array.coords['x'].to_dataset(name='X')
        expected = array + ds
        actual = grouped + ds
        assert_identical(expected, actual)
        actual = ds + grouped
        assert_identical(expected, actual)

    def test_groupby_math(self) -> None:
        array = self.da
        grouped = array.groupby('abc')
        expected_agg = (grouped.mean(...) - np.arange(3)).rename(None)
        actual = grouped - DataArray(range(3), [('abc', ['a', 'b', 'c'])])
        actual_agg = actual.groupby('abc').mean(...)
        assert_allclose(expected_agg, actual_agg)
        with pytest.raises(TypeError, match='only support binary ops'):
            grouped + 1
        with pytest.raises(TypeError, match='only support binary ops'):
            grouped + grouped
        with pytest.raises(TypeError, match='in-place operations'):
            array += grouped

    def test_groupby_math_not_aligned(self) -> None:
        array = DataArray(range(4), {'b': ('x', [0, 0, 1, 1]), 'x': [0, 1, 2, 3]}, dims='x')
        other = DataArray([10], coords={'b': [0]}, dims='b')
        actual = array.groupby('b') + other
        expected = DataArray([10, 11, np.nan, np.nan], array.coords)
        assert_identical(expected, actual)
        other = array.groupby('b').sum()
        actual = array.sel(x=[0, 1]).groupby('b') - other
        expected = DataArray([-1, 0], {'b': ('x', [0, 0]), 'x': [0, 1]}, dims='x')
        assert_identical(expected, actual)
        other = DataArray([10], coords={'c': 123, 'b': [0]}, dims='b')
        actual = array.groupby('b') + other
        expected = DataArray([10, 11, np.nan, np.nan], array.coords)
        expected.coords['c'] = (['x'], [123] * 2 + [np.nan] * 2)
        assert_identical(expected, actual)
        other_ds = Dataset({'a': ('b', [10])}, {'b': [0]})
        actual_ds = array.groupby('b') + other_ds
        expected_ds = Dataset({'a': ('x', [10, 11, np.nan, np.nan])}, array.coords)
        assert_identical(expected_ds, actual_ds)

    def test_groupby_restore_dim_order(self) -> None:
        array = DataArray(np.random.randn(5, 3), coords={'a': ('x', range(5)), 'b': ('y', range(3))}, dims=['x', 'y'])
        for by, expected_dims in [('x', ('x', 'y')), ('y', ('x', 'y')), ('a', ('a', 'y')), ('b', ('x', 'b'))]:
            result = array.groupby(by, squeeze=False).map(lambda x: x.squeeze())
            assert result.dims == expected_dims

    def test_groupby_restore_coord_dims(self) -> None:
        array = DataArray(np.random.randn(5, 3), coords={'a': ('x', range(5)), 'b': ('y', range(3)), 'c': (('x', 'y'), np.random.randn(5, 3))}, dims=['x', 'y'])
        for by, expected_dims in [('x', ('x', 'y')), ('y', ('x', 'y')), ('a', ('a', 'y')), ('b', ('x', 'b'))]:
            result = array.groupby(by, squeeze=False, restore_coord_dims=True).map(lambda x: x.squeeze())['c']
            assert result.dims == expected_dims

    def test_groupby_first_and_last(self) -> None:
        array = DataArray([1, 2, 3, 4, 5], dims='x')
        by = DataArray(['a'] * 2 + ['b'] * 3, dims='x', name='ab')
        expected = DataArray([1, 3], [('ab', ['a', 'b'])])
        actual = array.groupby(by).first()
        assert_identical(expected, actual)
        expected = DataArray([2, 5], [('ab', ['a', 'b'])])
        actual = array.groupby(by).last()
        assert_identical(expected, actual)
        array = DataArray(np.random.randn(5, 3), dims=['x', 'y'])
        expected = DataArray(array[[0, 2]], {'ab': ['a', 'b']}, ['ab', 'y'])
        actual = array.groupby(by).first()
        assert_identical(expected, actual)
        actual = array.groupby('x').first()
        expected = array
        assert_identical(expected, actual)

    def make_groupby_multidim_example_array(self) -> DataArray:
        return DataArray([[[0, 1], [2, 3]], [[5, 10], [15, 20]]], coords={'lon': (['ny', 'nx'], [[30, 40], [40, 50]]), 'lat': (['ny', 'nx'], [[10, 10], [20, 20]])}, dims=['time', 'ny', 'nx'])

    def test_groupby_multidim(self) -> None:
        array = self.make_groupby_multidim_example_array()
        for dim, expected_sum in [('lon', DataArray([5, 28, 23], coords=[('lon', [30.0, 40.0, 50.0])])), ('lat', DataArray([16, 40], coords=[('lat', [10.0, 20.0])]))]:
            actual_sum = array.groupby(dim).sum(...)
            assert_identical(expected_sum, actual_sum)

    def test_groupby_multidim_map(self) -> None:
        array = self.make_groupby_multidim_example_array()
        actual = array.groupby('lon').map(lambda x: x - x.mean())
        expected = DataArray([[[-2.5, -6.0], [-5.0, -8.5]], [[2.5, 3.0], [8.0, 8.5]]], coords=array.coords, dims=array.dims)
        assert_identical(expected, actual)

    @pytest.mark.parametrize('use_flox', [True, False])
    @pytest.mark.parametrize('coords', [np.arange(4), np.arange(4)[::-1], [2, 0, 3, 1]])
    @pytest.mark.parametrize('cut_kwargs', ({'labels': None, 'include_lowest': True}, {'labels': None, 'include_lowest': False}, {'labels': ['a', 'b']}, {'labels': [1.2, 3.5]}, {'labels': ['b', 'a']}))
    def test_groupby_bins(self, coords: np.typing.ArrayLike, use_flox: bool, cut_kwargs: dict) -> None:
        array = DataArray(np.arange(4), dims='dim_0', coords={'dim_0': coords}, name='a')
        array[0] = 99
        bins = [0, 1.5, 5]
        df = array.to_dataframe()
        df['dim_0_bins'] = pd.cut(array['dim_0'], bins, **cut_kwargs)
        expected_df = df.groupby('dim_0_bins', observed=True).sum()
        expected = expected_df.reset_index(drop=True).to_xarray().assign_coords(index=np.array(expected_df.index)).rename({'index': 'dim_0_bins'})['a']
        with xr.set_options(use_flox=use_flox):
            actual = array.groupby_bins('dim_0', bins=bins, **cut_kwargs).sum()
            assert_identical(expected, actual)
            actual = array.groupby_bins('dim_0', bins=bins, **cut_kwargs).map(lambda x: x.sum())
            assert_identical(expected, actual)
            assert len(array.dim_0) == 4

    def test_groupby_bins_ellipsis(self) -> None:
        da = xr.DataArray(np.ones((2, 3, 4)))
        bins = [-1, 0, 1, 2]
        with xr.set_options(use_flox=False):
            actual = da.groupby_bins('dim_0', bins).mean(...)
        with xr.set_options(use_flox=True):
            expected = da.groupby_bins('dim_0', bins).mean(...)
        assert_allclose(actual, expected)

    @pytest.mark.parametrize('use_flox', [True, False])
    def test_groupby_bins_gives_correct_subset(self, use_flox: bool) -> None:
        rng = np.random.default_rng(42)
        coords = rng.normal(5, 5, 1000)
        bins = np.logspace(-4, 1, 10)
        labels = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        darr = xr.DataArray(coords, coords=[coords], dims=['coords'])
        expected = xr.DataArray([np.nan, np.nan, 1, 1, 1, 8, 31, 104, 542], dims='coords_bins', coords={'coords_bins': labels})
        gb = darr.groupby_bins('coords', bins, labels=labels)
        with xr.set_options(use_flox=use_flox):
            actual = gb.count()
        assert_identical(actual, expected)

    def test_groupby_bins_empty(self) -> None:
        array = DataArray(np.arange(4), [('x', range(4))])
        bins = [0, 4, 5]
        bin_coords = pd.cut(array['x'], bins).categories
        actual = array.groupby_bins('x', bins).sum()
        expected = DataArray([6, np.nan], dims='x_bins', coords={'x_bins': bin_coords})
        assert_identical(expected, actual)
        assert len(array.x) == 4

    def test_groupby_bins_multidim(self) -> None:
        array = self.make_groupby_multidim_example_array()
        bins = [0, 15, 20]
        bin_coords = pd.cut(array['lat'].values.flat, bins).categories
        expected = DataArray([16, 40], dims='lat_bins', coords={'lat_bins': bin_coords})
        actual = array.groupby_bins('lat', bins).map(lambda x: x.sum())
        assert_identical(expected, actual)
        array['lat'].data = np.array([[10.0, 20.0], [20.0, 10.0]])
        expected = DataArray([28, 28], dims='lat_bins', coords={'lat_bins': bin_coords})
        actual = array.groupby_bins('lat', bins).map(lambda x: x.sum())
        assert_identical(expected, actual)
        bins = [-2, -1, 0, 1, 2]
        field = DataArray(np.ones((5, 3)), dims=('x', 'y'))
        by = DataArray(np.array([[-1.5, -1.5, 0.5, 1.5, 1.5] * 3]).reshape(5, 3), dims=('x', 'y'))
        actual = field.groupby_bins(by, bins=bins).count()
        bincoord = np.array([pd.Interval(left, right, closed='right') for left, right in zip(bins[:-1], bins[1:])], dtype=object)
        expected = DataArray(np.array([6, np.nan, 3, 6]), dims='group_bins', coords={'group_bins': bincoord})
        assert_identical(actual, expected)

    def test_groupby_bins_sort(self) -> None:
        data = xr.DataArray(np.arange(100), dims='x', coords={'x': np.linspace(-100, 100, num=100)})
        binned_mean = data.groupby_bins('x', bins=11).mean()
        assert binned_mean.to_index().is_monotonic_increasing
        with xr.set_options(use_flox=True):
            actual = data.groupby_bins('x', bins=11).count()
        with xr.set_options(use_flox=False):
            expected = data.groupby_bins('x', bins=11).count()
        assert_identical(actual, expected)

    def test_groupby_assign_coords(self) -> None:
        array = DataArray([1, 2, 3, 4], {'c': ('x', [0, 0, 1, 1])}, dims='x')
        actual = array.groupby('c').assign_coords(d=lambda a: a.mean())
        expected = array.copy()
        expected.coords['d'] = ('x', [1.5, 1.5, 3.5, 3.5])
        assert_identical(actual, expected)

    def test_groupby_fillna(self) -> None:
        a = DataArray([np.nan, 1, np.nan, 3], coords={'x': range(4)}, dims='x')
        fill_value = DataArray([0, 1], dims='y')
        actual = a.fillna(fill_value)
        expected = DataArray([[0, 1], [1, 1], [0, 1], [3, 3]], coords={'x': range(4)}, dims=('x', 'y'))
        assert_identical(expected, actual)
        b = DataArray(range(4), coords={'x': range(4)}, dims='x')
        expected = b.copy()
        for target in [a, expected]:
            target.coords['b'] = ('x', [0, 0, 1, 1])
        actual = a.groupby('b').fillna(DataArray([0, 2], dims='b'))
        assert_identical(expected, actual)