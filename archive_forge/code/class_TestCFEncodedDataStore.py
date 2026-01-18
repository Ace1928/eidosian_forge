from __future__ import annotations
import contextlib
import warnings
import numpy as np
import pandas as pd
import pytest
from xarray import (
from xarray.backends.common import WritableCFDataStore
from xarray.backends.memory import InMemoryDataStore
from xarray.conventions import decode_cf
from xarray.testing import assert_identical
from xarray.tests import (
from xarray.tests.test_backends import CFEncodedBase
@requires_netCDF4
class TestCFEncodedDataStore(CFEncodedBase):

    @contextlib.contextmanager
    def create_store(self):
        yield CFEncodedInMemoryStore()

    @contextlib.contextmanager
    def roundtrip(self, data, save_kwargs=None, open_kwargs=None, allow_cleanup_failure=False):
        if save_kwargs is None:
            save_kwargs = {}
        if open_kwargs is None:
            open_kwargs = {}
        store = CFEncodedInMemoryStore()
        data.dump_to_store(store, **save_kwargs)
        yield open_dataset(store, **open_kwargs)

    @pytest.mark.skip('cannot roundtrip coordinates yet for CFEncodedInMemoryStore')
    def test_roundtrip_coordinates(self) -> None:
        pass

    def test_invalid_dataarray_names_raise(self) -> None:
        pass

    def test_encoding_kwarg(self) -> None:
        pass

    def test_encoding_kwarg_fixed_width_string(self) -> None:
        pass