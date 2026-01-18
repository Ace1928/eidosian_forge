from __future__ import annotations
import pickle
import pytest
import xarray as xr
from xarray.tests import assert_identical
@xr.register_dataset_accessor('example_accessor')
@xr.register_dataarray_accessor('example_accessor')
class ExampleAccessor:
    """For the pickling tests below."""

    def __init__(self, xarray_obj):
        self.obj = xarray_obj