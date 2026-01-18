from __future__ import annotations
import pickle
import pytest
import xarray as xr
from xarray.tests import assert_identical
@xr.register_dataset_accessor('stupid_accessor')
class BrokenAccessor:

    def __init__(self, xarray_obj):
        raise AttributeError('broken')