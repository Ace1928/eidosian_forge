from __future__ import annotations
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes, merge
from xarray.core.merge import MergeError
from xarray.testing import assert_equal, assert_identical
from xarray.tests.test_dataset import create_test_data
class TestMergeFunction:

    def test_merge_arrays(self):
        data = create_test_data(add_attrs=False)
        actual = xr.merge([data.var1, data.var2])
        expected = data[['var1', 'var2']]
        assert_identical(actual, expected)

    def test_merge_datasets(self):
        data = create_test_data(add_attrs=False)
        actual = xr.merge([data[['var1']], data[['var2']]])
        expected = data[['var1', 'var2']]
        assert_identical(actual, expected)
        actual = xr.merge([data, data])
        assert_identical(actual, data)

    def test_merge_dataarray_unnamed(self):
        data = xr.DataArray([1, 2], dims='x')
        with pytest.raises(ValueError, match='without providing an explicit name'):
            xr.merge([data])

    def test_merge_arrays_attrs_default(self):
        var1_attrs = {'a': 1, 'b': 2}
        var2_attrs = {'a': 1, 'c': 3}
        expected_attrs = {'a': 1, 'b': 2}
        data = create_test_data(add_attrs=False)
        expected = data[['var1', 'var2']].copy()
        expected.var1.attrs = var1_attrs
        expected.var2.attrs = var2_attrs
        expected.attrs = expected_attrs
        data.var1.attrs = var1_attrs
        data.var2.attrs = var2_attrs
        actual = xr.merge([data.var1, data.var2])
        assert_identical(actual, expected)

    @pytest.mark.parametrize('combine_attrs, var1_attrs, var2_attrs, expected_attrs, expect_exception', [('no_conflicts', {'a': 1, 'b': 2}, {'a': 1, 'c': 3}, {'a': 1, 'b': 2, 'c': 3}, False), ('no_conflicts', {'a': 1, 'b': 2}, {}, {'a': 1, 'b': 2}, False), ('no_conflicts', {}, {'a': 1, 'c': 3}, {'a': 1, 'c': 3}, False), ('no_conflicts', {'a': 1, 'b': 2}, {'a': 4, 'c': 3}, {'a': 1, 'b': 2, 'c': 3}, True), ('drop', {'a': 1, 'b': 2}, {'a': 1, 'c': 3}, {}, False), ('identical', {'a': 1, 'b': 2}, {'a': 1, 'b': 2}, {'a': 1, 'b': 2}, False), ('identical', {'a': 1, 'b': 2}, {'a': 1, 'c': 3}, {'a': 1, 'b': 2}, True), ('override', {'a': 1, 'b': 2}, {'a': 4, 'b': 5, 'c': 3}, {'a': 1, 'b': 2}, False), ('drop_conflicts', {'a': 1, 'b': 2, 'c': 3}, {'b': 1, 'c': 3, 'd': 4}, {'a': 1, 'c': 3, 'd': 4}, False), ('drop_conflicts', {'a': 1, 'b': np.array([2]), 'c': np.array([3])}, {'b': 1, 'c': np.array([3]), 'd': 4}, {'a': 1, 'c': np.array([3]), 'd': 4}, False), (lambda attrs, context: attrs[1], {'a': 1, 'b': 2, 'c': 3}, {'a': 4, 'b': 3, 'c': 1}, {'a': 4, 'b': 3, 'c': 1}, False)])
    def test_merge_arrays_attrs(self, combine_attrs, var1_attrs, var2_attrs, expected_attrs, expect_exception):
        data1 = xr.Dataset(attrs=var1_attrs)
        data2 = xr.Dataset(attrs=var2_attrs)
        if expect_exception:
            with pytest.raises(MergeError, match='combine_attrs'):
                actual = xr.merge([data1, data2], combine_attrs=combine_attrs)
        else:
            actual = xr.merge([data1, data2], combine_attrs=combine_attrs)
            expected = xr.Dataset(attrs=expected_attrs)
            assert_identical(actual, expected)

    @pytest.mark.parametrize('combine_attrs, attrs1, attrs2, expected_attrs, expect_exception', [('no_conflicts', {'a': 1, 'b': 2}, {'a': 1, 'c': 3}, {'a': 1, 'b': 2, 'c': 3}, False), ('no_conflicts', {'a': 1, 'b': 2}, {}, {'a': 1, 'b': 2}, False), ('no_conflicts', {}, {'a': 1, 'c': 3}, {'a': 1, 'c': 3}, False), ('no_conflicts', {'a': 1, 'b': 2}, {'a': 4, 'c': 3}, {'a': 1, 'b': 2, 'c': 3}, True), ('drop', {'a': 1, 'b': 2}, {'a': 1, 'c': 3}, {}, False), ('identical', {'a': 1, 'b': 2}, {'a': 1, 'b': 2}, {'a': 1, 'b': 2}, False), ('identical', {'a': 1, 'b': 2}, {'a': 1, 'c': 3}, {'a': 1, 'b': 2}, True), ('override', {'a': 1, 'b': 2}, {'a': 4, 'b': 5, 'c': 3}, {'a': 1, 'b': 2}, False), ('drop_conflicts', {'a': 1, 'b': 2, 'c': 3}, {'b': 1, 'c': 3, 'd': 4}, {'a': 1, 'c': 3, 'd': 4}, False), (lambda attrs, context: attrs[1], {'a': 1, 'b': 2, 'c': 3}, {'a': 4, 'b': 3, 'c': 1}, {'a': 4, 'b': 3, 'c': 1}, False)])
    def test_merge_arrays_attrs_variables(self, combine_attrs, attrs1, attrs2, expected_attrs, expect_exception):
        """check that combine_attrs is used on data variables and coords"""
        data1 = xr.Dataset({'var1': ('dim1', [], attrs1)}, coords={'dim1': ('dim1', [], attrs1)})
        data2 = xr.Dataset({'var1': ('dim1', [], attrs2)}, coords={'dim1': ('dim1', [], attrs2)})
        if expect_exception:
            with pytest.raises(MergeError, match='combine_attrs'):
                actual = xr.merge([data1, data2], combine_attrs=combine_attrs)
        else:
            actual = xr.merge([data1, data2], combine_attrs=combine_attrs)
            expected = xr.Dataset({'var1': ('dim1', [], expected_attrs)}, coords={'dim1': ('dim1', [], expected_attrs)})
            assert_identical(actual, expected)

    def test_merge_attrs_override_copy(self):
        ds1 = xr.Dataset(attrs={'x': 0})
        ds2 = xr.Dataset(attrs={'x': 1})
        ds3 = xr.merge([ds1, ds2], combine_attrs='override')
        ds3.attrs['x'] = 2
        assert ds1.x == 0

    def test_merge_attrs_drop_conflicts(self):
        ds1 = xr.Dataset(attrs={'a': 0, 'b': 0, 'c': 0})
        ds2 = xr.Dataset(attrs={'b': 0, 'c': 1, 'd': 0})
        ds3 = xr.Dataset(attrs={'a': 0, 'b': 1, 'c': 0, 'e': 0})
        actual = xr.merge([ds1, ds2, ds3], combine_attrs='drop_conflicts')
        expected = xr.Dataset(attrs={'a': 0, 'd': 0, 'e': 0})
        assert_identical(actual, expected)

    def test_merge_attrs_no_conflicts_compat_minimal(self):
        """make sure compat="minimal" does not silence errors"""
        ds1 = xr.Dataset({'a': ('x', [], {'a': 0})})
        ds2 = xr.Dataset({'a': ('x', [], {'a': 1})})
        with pytest.raises(xr.MergeError, match='combine_attrs'):
            xr.merge([ds1, ds2], combine_attrs='no_conflicts', compat='minimal')

    def test_merge_dicts_simple(self):
        actual = xr.merge([{'foo': 0}, {'bar': 'one'}, {'baz': 3.5}])
        expected = xr.Dataset({'foo': 0, 'bar': 'one', 'baz': 3.5})
        assert_identical(actual, expected)

    def test_merge_dicts_dims(self):
        actual = xr.merge([{'y': ('x', [13])}, {'x': [12]}])
        expected = xr.Dataset({'x': [12], 'y': ('x', [13])})
        assert_identical(actual, expected)

    def test_merge_coordinates(self):
        coords1 = xr.Coordinates({'x': ('x', [0, 1, 2])})
        coords2 = xr.Coordinates({'y': ('y', [3, 4, 5])})
        expected = xr.Dataset(coords={'x': [0, 1, 2], 'y': [3, 4, 5]})
        actual = xr.merge([coords1, coords2])
        assert_identical(actual, expected)

    def test_merge_error(self):
        ds = xr.Dataset({'x': 0})
        with pytest.raises(xr.MergeError):
            xr.merge([ds, ds + 1])

    def test_merge_alignment_error(self):
        ds = xr.Dataset(coords={'x': [1, 2]})
        other = xr.Dataset(coords={'x': [2, 3]})
        with pytest.raises(ValueError, match='cannot align.*join.*exact.*not equal.*'):
            xr.merge([ds, other], join='exact')

    def test_merge_wrong_input_error(self):
        with pytest.raises(TypeError, match='objects must be an iterable'):
            xr.merge([1])
        ds = xr.Dataset(coords={'x': [1, 2]})
        with pytest.raises(TypeError, match='objects must be an iterable'):
            xr.merge({'a': ds})
        with pytest.raises(TypeError, match='objects must be an iterable'):
            xr.merge([ds, 1])

    def test_merge_no_conflicts_single_var(self):
        ds1 = xr.Dataset({'a': ('x', [1, 2]), 'x': [0, 1]})
        ds2 = xr.Dataset({'a': ('x', [2, 3]), 'x': [1, 2]})
        expected = xr.Dataset({'a': ('x', [1, 2, 3]), 'x': [0, 1, 2]})
        assert expected.identical(xr.merge([ds1, ds2], compat='no_conflicts'))
        assert expected.identical(xr.merge([ds2, ds1], compat='no_conflicts'))
        assert ds1.identical(xr.merge([ds1, ds2], compat='no_conflicts', join='left'))
        assert ds2.identical(xr.merge([ds1, ds2], compat='no_conflicts', join='right'))
        expected = xr.Dataset({'a': ('x', [2]), 'x': [1]})
        assert expected.identical(xr.merge([ds1, ds2], compat='no_conflicts', join='inner'))
        with pytest.raises(xr.MergeError):
            ds3 = xr.Dataset({'a': ('x', [99, 3]), 'x': [1, 2]})
            xr.merge([ds1, ds3], compat='no_conflicts')
        with pytest.raises(xr.MergeError):
            ds3 = xr.Dataset({'a': ('y', [2, 3]), 'y': [1, 2]})
            xr.merge([ds1, ds3], compat='no_conflicts')

    def test_merge_no_conflicts_multi_var(self):
        data = create_test_data(add_attrs=False)
        data1 = data.copy(deep=True)
        data2 = data.copy(deep=True)
        expected = data[['var1', 'var2']]
        actual = xr.merge([data1.var1, data2.var2], compat='no_conflicts')
        assert_identical(expected, actual)
        data1['var1'][:, :5] = np.nan
        data2['var1'][:, 5:] = np.nan
        data1['var2'][:4, :] = np.nan
        data2['var2'][4:, :] = np.nan
        del data2['var3']
        actual = xr.merge([data1, data2], compat='no_conflicts')
        assert_equal(data, actual)

    def test_merge_no_conflicts_preserve_attrs(self):
        data = xr.Dataset({'x': ([], 0, {'foo': 'bar'})})
        actual = xr.merge([data, data], combine_attrs='no_conflicts')
        assert_identical(data, actual)

    def test_merge_no_conflicts_broadcast(self):
        datasets = [xr.Dataset({'x': ('y', [0])}), xr.Dataset({'x': np.nan})]
        actual = xr.merge(datasets)
        expected = xr.Dataset({'x': ('y', [0])})
        assert_identical(expected, actual)
        datasets = [xr.Dataset({'x': ('y', [np.nan])}), xr.Dataset({'x': 0})]
        actual = xr.merge(datasets)
        assert_identical(expected, actual)