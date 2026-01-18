from __future__ import annotations
import pytest
import xarray
from xarray import concat, merge
from xarray.backends.file_manager import FILE_CACHE
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.tests.test_dataset import create_test_data
class TestAttrRetention:

    def test_dataset_attr_retention(self) -> None:
        ds = create_test_dataset_attrs()
        original_attrs = ds.attrs
        result = ds.mean()
        assert result.attrs == {}
        with xarray.set_options(keep_attrs='default'):
            result = ds.mean()
            assert result.attrs == {}
        with xarray.set_options(keep_attrs=True):
            result = ds.mean()
            assert result.attrs == original_attrs
        with xarray.set_options(keep_attrs=False):
            result = ds.mean()
            assert result.attrs == {}

    def test_dataarray_attr_retention(self) -> None:
        da = create_test_dataarray_attrs()
        original_attrs = da.attrs
        result = da.mean()
        assert result.attrs == {}
        with xarray.set_options(keep_attrs='default'):
            result = da.mean()
            assert result.attrs == {}
        with xarray.set_options(keep_attrs=True):
            result = da.mean()
            assert result.attrs == original_attrs
        with xarray.set_options(keep_attrs=False):
            result = da.mean()
            assert result.attrs == {}

    def test_groupby_attr_retention(self) -> None:
        da = xarray.DataArray([1, 2, 3], [('x', [1, 1, 2])])
        da.attrs = {'attr1': 5, 'attr2': 'history', 'attr3': {'nested': 'more_info'}}
        original_attrs = da.attrs
        result = da.groupby('x').sum(keep_attrs=True)
        assert result.attrs == original_attrs
        with xarray.set_options(keep_attrs='default'):
            result = da.groupby('x').sum(keep_attrs=True)
            assert result.attrs == original_attrs
        with xarray.set_options(keep_attrs=True):
            result1 = da.groupby('x')
            result = result1.sum()
            assert result.attrs == original_attrs
        with xarray.set_options(keep_attrs=False):
            result = da.groupby('x').sum()
            assert result.attrs == {}

    def test_concat_attr_retention(self) -> None:
        ds1 = create_test_dataset_attrs()
        ds2 = create_test_dataset_attrs()
        ds2.attrs = {'wrong': 'attributes'}
        original_attrs = ds1.attrs
        result = concat([ds1, ds2], dim='dim1')
        assert result.attrs == original_attrs

    def test_merge_attr_retention(self) -> None:
        da1 = create_test_dataarray_attrs(var='var1')
        da2 = create_test_dataarray_attrs(var='var2')
        da2.attrs = {'wrong': 'attributes'}
        original_attrs = da1.attrs
        result = merge([da1, da2])
        assert result.attrs == original_attrs

    def test_display_style_text(self) -> None:
        ds = create_test_dataset_attrs()
        with xarray.set_options(display_style='text'):
            text = ds._repr_html_()
            assert text.startswith('<pre>')
            assert '&#x27;nested&#x27;' in text
            assert '&lt;xarray.Dataset&gt;' in text

    def test_display_style_html(self) -> None:
        ds = create_test_dataset_attrs()
        with xarray.set_options(display_style='html'):
            html = ds._repr_html_()
            assert html.startswith('<div>')
            assert '&#x27;nested&#x27;' in html

    def test_display_dataarray_style_text(self) -> None:
        da = create_test_dataarray_attrs()
        with xarray.set_options(display_style='text'):
            text = da._repr_html_()
            assert text.startswith('<pre>')
            assert '&lt;xarray.DataArray &#x27;var1&#x27;' in text

    def test_display_dataarray_style_html(self) -> None:
        da = create_test_dataarray_attrs()
        with xarray.set_options(display_style='html'):
            html = da._repr_html_()
            assert html.startswith('<div>')
            assert '#x27;nested&#x27;' in html