from __future__ import annotations
import pickle
import pytest
import xarray as xr
from xarray.tests import assert_identical
class TestAccessor:

    def test_register(self) -> None:

        @xr.register_dataset_accessor('demo')
        @xr.register_dataarray_accessor('demo')
        class DemoAccessor:
            """Demo accessor."""

            def __init__(self, xarray_obj):
                self._obj = xarray_obj

            @property
            def foo(self):
                return 'bar'
        ds = xr.Dataset()
        assert ds.demo.foo == 'bar'
        da = xr.DataArray(0)
        assert da.demo.foo == 'bar'
        assert ds.demo is ds.demo
        assert ds.demo.__doc__ == 'Demo accessor.'
        assert xr.Dataset.demo.__doc__ == 'Demo accessor.'
        assert isinstance(ds.demo, DemoAccessor)
        assert xr.Dataset.demo is DemoAccessor
        del xr.Dataset.demo
        assert not hasattr(xr.Dataset, 'demo')
        with pytest.warns(Warning, match='overriding a preexisting attribute'):

            @xr.register_dataarray_accessor('demo')
            class Foo:
                pass
        assert not hasattr(xr.Dataset, 'demo')

    def test_pickle_dataset(self) -> None:
        ds = xr.Dataset()
        ds_restored = pickle.loads(pickle.dumps(ds))
        assert_identical(ds, ds_restored)
        assert ds.example_accessor is ds.example_accessor
        ds.example_accessor.value = 'foo'
        ds_restored = pickle.loads(pickle.dumps(ds))
        assert_identical(ds, ds_restored)
        assert ds_restored.example_accessor.value == 'foo'

    def test_pickle_dataarray(self) -> None:
        array = xr.Dataset()
        assert array.example_accessor is array.example_accessor
        array_restored = pickle.loads(pickle.dumps(array))
        assert_identical(array, array_restored)

    def test_broken_accessor(self) -> None:

        @xr.register_dataset_accessor('stupid_accessor')
        class BrokenAccessor:

            def __init__(self, xarray_obj):
                raise AttributeError('broken')
        with pytest.raises(RuntimeError, match='error initializing'):
            xr.Dataset().stupid_accessor