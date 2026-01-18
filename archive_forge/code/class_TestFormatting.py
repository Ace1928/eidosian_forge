from __future__ import annotations
import sys
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
from numpy.core import defchararray
import xarray as xr
from xarray.core import formatting
from xarray.tests import requires_cftime, requires_dask, requires_netCDF4
class TestFormatting:

    def test_get_indexer_at_least_n_items(self) -> None:
        cases = [((20,), (slice(10),), (slice(-10, None),)), ((3, 20), (0, slice(10)), (-1, slice(-10, None))), ((2, 10), (0, slice(10)), (-1, slice(-10, None))), ((2, 5), (slice(2), slice(None)), (slice(-2, None), slice(None))), ((1, 2, 5), (0, slice(2), slice(None)), (-1, slice(-2, None), slice(None))), ((2, 3, 5), (0, slice(2), slice(None)), (-1, slice(-2, None), slice(None))), ((1, 10, 1), (0, slice(10), slice(None)), (-1, slice(-10, None), slice(None))), ((2, 5, 1), (slice(2), slice(None), slice(None)), (slice(-2, None), slice(None), slice(None))), ((2, 5, 3), (0, slice(4), slice(None)), (-1, slice(-4, None), slice(None))), ((2, 3, 3), (slice(2), slice(None), slice(None)), (slice(-2, None), slice(None), slice(None)))]
        for shape, start_expected, end_expected in cases:
            actual = formatting._get_indexer_at_least_n_items(shape, 10, from_end=False)
            assert start_expected == actual
            actual = formatting._get_indexer_at_least_n_items(shape, 10, from_end=True)
            assert end_expected == actual

    def test_first_n_items(self) -> None:
        array = np.arange(100).reshape(10, 5, 2)
        for n in [3, 10, 13, 100, 200]:
            actual = formatting.first_n_items(array, n)
            expected = array.flat[:n]
            assert (expected == actual).all()
        with pytest.raises(ValueError, match='at least one item'):
            formatting.first_n_items(array, 0)

    def test_last_n_items(self) -> None:
        array = np.arange(100).reshape(10, 5, 2)
        for n in [3, 10, 13, 100, 200]:
            actual = formatting.last_n_items(array, n)
            expected = array.flat[-n:]
            assert (expected == actual).all()
        with pytest.raises(ValueError, match='at least one item'):
            formatting.first_n_items(array, 0)

    def test_last_item(self) -> None:
        array = np.arange(100)
        reshape = ((10, 10), (1, 100), (2, 2, 5, 5))
        expected = np.array([99])
        for r in reshape:
            result = formatting.last_item(array.reshape(r))
            assert result == expected

    def test_format_item(self) -> None:
        cases = [(pd.Timestamp('2000-01-01T12'), '2000-01-01T12:00:00'), (pd.Timestamp('2000-01-01'), '2000-01-01'), (pd.Timestamp('NaT'), 'NaT'), (pd.Timedelta('10 days 1 hour'), '10 days 01:00:00'), (pd.Timedelta('-3 days'), '-3 days +00:00:00'), (pd.Timedelta('3 hours'), '0 days 03:00:00'), (pd.Timedelta('NaT'), 'NaT'), ('foo', "'foo'"), (b'foo', "b'foo'"), (1, '1'), (1.0, '1.0'), (np.float16(1.1234), '1.123'), (np.float32(1.0111111), '1.011'), (np.float64(22.222222), '22.22')]
        for item, expected in cases:
            actual = formatting.format_item(item)
            assert expected == actual

    def test_format_items(self) -> None:
        cases = [(np.arange(4) * np.timedelta64(1, 'D'), '0 days 1 days 2 days 3 days'), (np.arange(4) * np.timedelta64(3, 'h'), '00:00:00 03:00:00 06:00:00 09:00:00'), (np.arange(4) * np.timedelta64(500, 'ms'), '00:00:00 00:00:00.500000 00:00:01 00:00:01.500000'), (pd.to_timedelta(['NaT', '0s', '1s', 'NaT']), 'NaT 00:00:00 00:00:01 NaT'), (pd.to_timedelta(['1 day 1 hour', '1 day', '0 hours']), '1 days 01:00:00 1 days 00:00:00 0 days 00:00:00'), ([1, 2, 3], '1 2 3')]
        for item, expected in cases:
            actual = ' '.join(formatting.format_items(item))
            assert expected == actual

    def test_format_array_flat(self) -> None:
        actual = formatting.format_array_flat(np.arange(100), 2)
        expected = '...'
        assert expected == actual
        actual = formatting.format_array_flat(np.arange(100), 9)
        expected = '0 ... 99'
        assert expected == actual
        actual = formatting.format_array_flat(np.arange(100), 10)
        expected = '0 1 ... 99'
        assert expected == actual
        actual = formatting.format_array_flat(np.arange(100), 13)
        expected = '0 1 ... 98 99'
        assert expected == actual
        actual = formatting.format_array_flat(np.arange(100), 15)
        expected = '0 1 2 ... 98 99'
        assert expected == actual
        actual = formatting.format_array_flat(np.arange(100.0), 11)
        expected = '0.0 ... ...'
        assert expected == actual
        actual = formatting.format_array_flat(np.arange(100.0), 12)
        expected = '0.0 ... 99.0'
        assert expected == actual
        actual = formatting.format_array_flat(np.arange(3), 5)
        expected = '0 1 2'
        assert expected == actual
        actual = formatting.format_array_flat(np.arange(4.0), 11)
        expected = '0.0 ... 3.0'
        assert expected == actual
        actual = formatting.format_array_flat(np.arange(0), 0)
        expected = ''
        assert expected == actual
        actual = formatting.format_array_flat(np.arange(1), 1)
        expected = '0'
        assert expected == actual
        actual = formatting.format_array_flat(np.arange(2), 3)
        expected = '0 1'
        assert expected == actual
        actual = formatting.format_array_flat(np.arange(4), 7)
        expected = '0 1 2 3'
        assert expected == actual
        actual = formatting.format_array_flat(np.arange(5), 7)
        expected = '0 ... 4'
        assert expected == actual
        long_str = [' '.join(['hello world' for _ in range(100)])]
        actual = formatting.format_array_flat(np.asarray([long_str]), 21)
        expected = "'hello world hello..."
        assert expected == actual

    def test_pretty_print(self) -> None:
        assert formatting.pretty_print('abcdefghij', 8) == 'abcde...'
        assert formatting.pretty_print('ß', 1) == 'ß'

    def test_maybe_truncate(self) -> None:
        assert formatting.maybe_truncate('ß', 10) == 'ß'

    def test_format_timestamp_invalid_pandas_format(self) -> None:
        expected = '2021-12-06 17:00:00 00'
        with pytest.raises(ValueError):
            formatting.format_timestamp(expected)

    def test_format_timestamp_out_of_bounds(self) -> None:
        from datetime import datetime
        date = datetime(1300, 12, 1)
        expected = '1300-12-01'
        result = formatting.format_timestamp(date)
        assert result == expected
        date = datetime(2300, 12, 1)
        expected = '2300-12-01'
        result = formatting.format_timestamp(date)
        assert result == expected

    def test_attribute_repr(self) -> None:
        short = formatting.summarize_attr('key', 'Short string')
        long = formatting.summarize_attr('key', 100 * 'Very long string ')
        newlines = formatting.summarize_attr('key', '\n\n\n')
        tabs = formatting.summarize_attr('key', '\t\t\t')
        assert short == '    key: Short string'
        assert len(long) <= 80
        assert long.endswith('...')
        assert '\n' not in newlines
        assert '\t' not in tabs

    def test_index_repr(self) -> None:
        from xarray.core.indexes import Index

        class CustomIndex(Index):
            names: tuple[str, ...]

            def __init__(self, names: tuple[str, ...]):
                self.names = names

            def __repr__(self):
                return f'CustomIndex(coords={self.names})'
        coord_names = ('x', 'y')
        index = CustomIndex(coord_names)
        names = ('x',)
        normal = formatting.summarize_index(names, index, col_width=20)
        assert names[0] in normal
        assert len(normal.splitlines()) == len(names)
        assert 'CustomIndex' in normal

        class IndexWithInlineRepr(CustomIndex):

            def _repr_inline_(self, max_width: int):
                return f'CustomIndex[{', '.join(self.names)}]'
        index = IndexWithInlineRepr(coord_names)
        inline = formatting.summarize_index(names, index, col_width=20)
        assert names[0] in inline
        assert index._repr_inline_(max_width=40) in inline

    @pytest.mark.parametrize('names', (('x',), ('x', 'y'), ('x', 'y', 'z'), ('x', 'y', 'z', 'a')))
    def test_index_repr_grouping(self, names) -> None:
        from xarray.core.indexes import Index

        class CustomIndex(Index):

            def __init__(self, names):
                self.names = names

            def __repr__(self):
                return f'CustomIndex(coords={self.names})'
        index = CustomIndex(names)
        normal = formatting.summarize_index(names, index, col_width=20)
        assert all((name in normal for name in names))
        assert len(normal.splitlines()) == len(names)
        assert 'CustomIndex' in normal
        hint_chars = [line[2] for line in normal.splitlines()]
        if len(names) <= 1:
            assert hint_chars == [' ']
        else:
            assert hint_chars[0] == '┌' and hint_chars[-1] == '└'
            assert len(names) == 2 or hint_chars[1:-1] == ['│'] * (len(names) - 2)

    def test_diff_array_repr(self) -> None:
        da_a = xr.DataArray(np.array([[1, 2, 3], [4, 5, 6]], dtype='int64'), dims=('x', 'y'), coords={'x': np.array(['a', 'b'], dtype='U1'), 'y': np.array([1, 2, 3], dtype='int64')}, attrs={'units': 'm', 'description': 'desc'})
        da_b = xr.DataArray(np.array([1, 2], dtype='int64'), dims='x', coords={'x': np.array(['a', 'c'], dtype='U1'), 'label': ('x', np.array([1, 2], dtype='int64'))}, attrs={'units': 'kg'})
        byteorder = '<' if sys.byteorder == 'little' else '>'
        expected = dedent("        Left and right DataArray objects are not identical\n        Differing dimensions:\n            (x: 2, y: 3) != (x: 2)\n        Differing values:\n        L\n            array([[1, 2, 3],\n                   [4, 5, 6]], dtype=int64)\n        R\n            array([1, 2], dtype=int64)\n        Differing coordinates:\n        L * x        (x) %cU1 8B 'a' 'b'\n        R * x        (x) %cU1 8B 'a' 'c'\n        Coordinates only on the left object:\n          * y        (y) int64 24B 1 2 3\n        Coordinates only on the right object:\n            label    (x) int64 16B 1 2\n        Differing attributes:\n        L   units: m\n        R   units: kg\n        Attributes only on the left object:\n            description: desc" % (byteorder, byteorder))
        actual = formatting.diff_array_repr(da_a, da_b, 'identical')
        try:
            assert actual == expected
        except AssertionError:
            assert actual == expected.replace(', dtype=int64', '')
        va = xr.Variable('x', np.array([1, 2, 3], dtype='int64'), {'title': 'test Variable'})
        vb = xr.Variable(('x', 'y'), np.array([[1, 2, 3], [4, 5, 6]], dtype='int64'))
        expected = dedent('        Left and right Variable objects are not equal\n        Differing dimensions:\n            (x: 3) != (x: 2, y: 3)\n        Differing values:\n        L\n            array([1, 2, 3], dtype=int64)\n        R\n            array([[1, 2, 3],\n                   [4, 5, 6]], dtype=int64)')
        actual = formatting.diff_array_repr(va, vb, 'equals')
        try:
            assert actual == expected
        except AssertionError:
            assert actual == expected.replace(', dtype=int64', '')

    @pytest.mark.filterwarnings('error')
    def test_diff_attrs_repr_with_array(self) -> None:
        attrs_a = {'attr': np.array([0, 1])}
        attrs_b = {'attr': 1}
        expected = dedent('            Differing attributes:\n            L   attr: [0 1]\n            R   attr: 1\n            ').strip()
        actual = formatting.diff_attrs_repr(attrs_a, attrs_b, 'equals')
        assert expected == actual
        attrs_c = {'attr': np.array([-3, 5])}
        expected = dedent('            Differing attributes:\n            L   attr: [0 1]\n            R   attr: [-3  5]\n            ').strip()
        actual = formatting.diff_attrs_repr(attrs_a, attrs_c, 'equals')
        assert expected == actual
        attrs_c = {'attr': np.array([0, 1, 2])}
        expected = dedent('            Differing attributes:\n            L   attr: [0 1]\n            R   attr: [0 1 2]\n            ').strip()
        actual = formatting.diff_attrs_repr(attrs_a, attrs_c, 'equals')
        assert expected == actual

    def test_diff_dataset_repr(self) -> None:
        ds_a = xr.Dataset(data_vars={'var1': (('x', 'y'), np.array([[1, 2, 3], [4, 5, 6]], dtype='int64')), 'var2': ('x', np.array([3, 4], dtype='int64'))}, coords={'x': ('x', np.array(['a', 'b'], dtype='U1'), {'foo': 'bar', 'same': 'same'}), 'y': np.array([1, 2, 3], dtype='int64')}, attrs={'title': 'mytitle', 'description': 'desc'})
        ds_b = xr.Dataset(data_vars={'var1': ('x', np.array([1, 2], dtype='int64'))}, coords={'x': ('x', np.array(['a', 'c'], dtype='U1'), {'source': 0, 'foo': 'baz', 'same': 'same'}), 'label': ('x', np.array([1, 2], dtype='int64'))}, attrs={'title': 'newtitle'})
        byteorder = '<' if sys.byteorder == 'little' else '>'
        expected = dedent("        Left and right Dataset objects are not identical\n        Differing dimensions:\n            (x: 2, y: 3) != (x: 2)\n        Differing coordinates:\n        L * x        (x) %cU1 8B 'a' 'b'\n            Differing variable attributes:\n                foo: bar\n        R * x        (x) %cU1 8B 'a' 'c'\n            Differing variable attributes:\n                source: 0\n                foo: baz\n        Coordinates only on the left object:\n          * y        (y) int64 24B 1 2 3\n        Coordinates only on the right object:\n            label    (x) int64 16B 1 2\n        Differing data variables:\n        L   var1     (x, y) int64 48B 1 2 3 4 5 6\n        R   var1     (x) int64 16B 1 2\n        Data variables only on the left object:\n            var2     (x) int64 16B 3 4\n        Differing attributes:\n        L   title: mytitle\n        R   title: newtitle\n        Attributes only on the left object:\n            description: desc" % (byteorder, byteorder))
        actual = formatting.diff_dataset_repr(ds_a, ds_b, 'identical')
        assert actual == expected

    def test_array_repr(self) -> None:
        ds = xr.Dataset(coords={'foo': np.array([1, 2, 3], dtype=np.uint64), 'bar': np.array([1, 2, 3], dtype=np.uint64)})
        ds[1, 2] = xr.DataArray(np.array([0], dtype=np.uint64), dims='test')
        ds_12 = ds[1, 2]
        actual = formatting.array_repr(ds_12)
        expected = dedent('        <xarray.DataArray (1, 2) (test: 1)> Size: 8B\n        array([0], dtype=uint64)\n        Dimensions without coordinates: test')
        assert actual == expected
        assert repr(ds_12) == expected
        assert str(ds_12) == expected
        actual = f'{ds_12}'
        assert actual == expected
        with xr.set_options(display_expand_data=False):
            actual = formatting.array_repr(ds[1, 2])
            expected = dedent('            <xarray.DataArray (1, 2) (test: 1)> Size: 8B\n            0\n            Dimensions without coordinates: test')
            assert actual == expected

    def test_array_repr_variable(self) -> None:
        var = xr.Variable('x', [0, 1])
        formatting.array_repr(var)
        with xr.set_options(display_expand_data=False):
            formatting.array_repr(var)

    def test_array_repr_recursive(self) -> None:
        var = xr.Variable('x', [0, 1])
        var.attrs['x'] = var
        formatting.array_repr(var)
        da = xr.DataArray([0, 1], dims=['x'])
        da.attrs['x'] = da
        formatting.array_repr(da)
        var.attrs['x'] = da
        da.attrs['x'] = var
        formatting.array_repr(var)
        formatting.array_repr(da)

    @requires_dask
    def test_array_scalar_format(self) -> None:
        var = xr.DataArray(np.array(0))
        assert format(var, '') == repr(var)
        assert format(var, 'd') == '0'
        assert format(var, '.2f') == '0.00'
        import dask.array as da
        var = xr.DataArray(da.array(0))
        assert format(var, '') == repr(var)
        with pytest.raises(TypeError) as excinfo:
            format(var, '.2f')
        assert 'unsupported format string passed to' in str(excinfo.value)
        var = xr.DataArray([0.1, 0.2])
        with pytest.raises(NotImplementedError) as excinfo:
            format(var, '.2f')
        assert 'Using format_spec is only supported' in str(excinfo.value)