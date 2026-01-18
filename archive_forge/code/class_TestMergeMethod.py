from __future__ import annotations
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes, merge
from xarray.core.merge import MergeError
from xarray.testing import assert_equal, assert_identical
from xarray.tests.test_dataset import create_test_data
class TestMergeMethod:

    def test_merge(self):
        data = create_test_data()
        ds1 = data[['var1']]
        ds2 = data[['var3']]
        expected = data[['var1', 'var3']]
        actual = ds1.merge(ds2)
        assert_identical(expected, actual)
        actual = ds2.merge(ds1)
        assert_identical(expected, actual)
        actual = data.merge(data)
        assert_identical(data, actual)
        actual = data.reset_coords(drop=True).merge(data)
        assert_identical(data, actual)
        actual = data.merge(data.reset_coords(drop=True))
        assert_identical(data, actual)
        with pytest.raises(ValueError):
            ds1.merge(ds2.rename({'var3': 'var1'}))
        with pytest.raises(ValueError, match='should be coordinates or not'):
            data.reset_coords().merge(data)
        with pytest.raises(ValueError, match='should be coordinates or not'):
            data.merge(data.reset_coords())

    def test_merge_broadcast_equals(self):
        ds1 = xr.Dataset({'x': 0})
        ds2 = xr.Dataset({'x': ('y', [0, 0])})
        actual = ds1.merge(ds2)
        assert_identical(ds2, actual)
        actual = ds2.merge(ds1)
        assert_identical(ds2, actual)
        actual = ds1.copy()
        actual.update(ds2)
        assert_identical(ds2, actual)
        ds1 = xr.Dataset({'x': np.nan})
        ds2 = xr.Dataset({'x': ('y', [np.nan, np.nan])})
        actual = ds1.merge(ds2)
        assert_identical(ds2, actual)

    def test_merge_compat(self):
        ds1 = xr.Dataset({'x': 0})
        ds2 = xr.Dataset({'x': 1})
        for compat in ['broadcast_equals', 'equals', 'identical', 'no_conflicts']:
            with pytest.raises(xr.MergeError):
                ds1.merge(ds2, compat=compat)
        ds2 = xr.Dataset({'x': [0, 0]})
        for compat in ['equals', 'identical']:
            with pytest.raises(ValueError, match='should be coordinates or not'):
                ds1.merge(ds2, compat=compat)
        ds2 = xr.Dataset({'x': ((), 0, {'foo': 'bar'})})
        with pytest.raises(xr.MergeError):
            ds1.merge(ds2, compat='identical')
        with pytest.raises(ValueError, match='compat=.* invalid'):
            ds1.merge(ds2, compat='foobar')
        assert ds1.identical(ds1.merge(ds2, compat='override'))

    def test_merge_compat_minimal(self) -> None:
        ds1 = xr.Dataset(coords={'foo': [1, 2, 3], 'bar': 4})
        ds2 = xr.Dataset(coords={'foo': [1, 2, 3], 'bar': 5})
        actual = xr.merge([ds1, ds2], compat='minimal')
        expected = xr.Dataset(coords={'foo': [1, 2, 3]})
        assert_identical(actual, expected)

    def test_merge_auto_align(self):
        ds1 = xr.Dataset({'a': ('x', [1, 2]), 'x': [0, 1]})
        ds2 = xr.Dataset({'b': ('x', [3, 4]), 'x': [1, 2]})
        expected = xr.Dataset({'a': ('x', [1, 2, np.nan]), 'b': ('x', [np.nan, 3, 4])}, {'x': [0, 1, 2]})
        assert expected.identical(ds1.merge(ds2))
        assert expected.identical(ds2.merge(ds1))
        expected = expected.isel(x=slice(2))
        assert expected.identical(ds1.merge(ds2, join='left'))
        assert expected.identical(ds2.merge(ds1, join='right'))
        expected = expected.isel(x=slice(1, 2))
        assert expected.identical(ds1.merge(ds2, join='inner'))
        assert expected.identical(ds2.merge(ds1, join='inner'))

    @pytest.mark.parametrize('fill_value', [dtypes.NA, 2, 2.0, {'a': 2, 'b': 1}])
    def test_merge_fill_value(self, fill_value):
        ds1 = xr.Dataset({'a': ('x', [1, 2]), 'x': [0, 1]})
        ds2 = xr.Dataset({'b': ('x', [3, 4]), 'x': [1, 2]})
        if fill_value == dtypes.NA:
            fill_value_a = fill_value_b = np.nan
        elif isinstance(fill_value, dict):
            fill_value_a = fill_value['a']
            fill_value_b = fill_value['b']
        else:
            fill_value_a = fill_value_b = fill_value
        expected = xr.Dataset({'a': ('x', [1, 2, fill_value_a]), 'b': ('x', [fill_value_b, 3, 4])}, {'x': [0, 1, 2]})
        assert expected.identical(ds1.merge(ds2, fill_value=fill_value))
        assert expected.identical(ds2.merge(ds1, fill_value=fill_value))
        assert expected.identical(xr.merge([ds1, ds2], fill_value=fill_value))

    def test_merge_no_conflicts(self):
        ds1 = xr.Dataset({'a': ('x', [1, 2]), 'x': [0, 1]})
        ds2 = xr.Dataset({'a': ('x', [2, 3]), 'x': [1, 2]})
        expected = xr.Dataset({'a': ('x', [1, 2, 3]), 'x': [0, 1, 2]})
        assert expected.identical(ds1.merge(ds2, compat='no_conflicts'))
        assert expected.identical(ds2.merge(ds1, compat='no_conflicts'))
        assert ds1.identical(ds1.merge(ds2, compat='no_conflicts', join='left'))
        assert ds2.identical(ds1.merge(ds2, compat='no_conflicts', join='right'))
        expected2 = xr.Dataset({'a': ('x', [2]), 'x': [1]})
        assert expected2.identical(ds1.merge(ds2, compat='no_conflicts', join='inner'))
        with pytest.raises(xr.MergeError):
            ds3 = xr.Dataset({'a': ('x', [99, 3]), 'x': [1, 2]})
            ds1.merge(ds3, compat='no_conflicts')
        with pytest.raises(xr.MergeError):
            ds3 = xr.Dataset({'a': ('y', [2, 3]), 'y': [1, 2]})
            ds1.merge(ds3, compat='no_conflicts')

    def test_merge_dataarray(self):
        ds = xr.Dataset({'a': 0})
        da = xr.DataArray(data=1, name='b')
        assert_identical(ds.merge(da), xr.merge([ds, da]))

    @pytest.mark.parametrize(['combine_attrs', 'attrs1', 'attrs2', 'expected_attrs', 'expect_error'], (('drop', {'a': 0, 'b': 1, 'c': 2}, {'a': 1, 'b': 2, 'c': 3}, {}, False), ('drop_conflicts', {'a': 0, 'b': 1, 'c': 2}, {'b': 2, 'c': 2, 'd': 3}, {'a': 0, 'c': 2, 'd': 3}, False), ('override', {'a': 0, 'b': 1}, {'a': 1, 'b': 2}, {'a': 0, 'b': 1}, False), ('no_conflicts', {'a': 0, 'b': 1}, {'a': 0, 'b': 2}, None, True), ('identical', {'a': 0, 'b': 1}, {'a': 0, 'b': 2}, None, True)))
    def test_merge_combine_attrs(self, combine_attrs, attrs1, attrs2, expected_attrs, expect_error):
        ds1 = xr.Dataset(attrs=attrs1)
        ds2 = xr.Dataset(attrs=attrs2)
        if expect_error:
            with pytest.raises(xr.MergeError):
                ds1.merge(ds2, combine_attrs=combine_attrs)
        else:
            actual = ds1.merge(ds2, combine_attrs=combine_attrs)
            expected = xr.Dataset(attrs=expected_attrs)
            assert_identical(actual, expected)