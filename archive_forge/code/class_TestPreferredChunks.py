from __future__ import annotations
from numbers import Number
import numpy as np
import pytest
import xarray as xr
from xarray.backends.api import _get_default_engine
from xarray.tests import (
@requires_dask
class TestPreferredChunks:
    """Test behaviors related to the backend's preferred chunks."""
    var_name = 'data'

    def create_dataset(self, shape, pref_chunks):
        """Return a dataset with a variable with the given shape and preferred chunks."""
        dims = tuple((f'dim_{idx}' for idx in range(len(shape))))
        return xr.Dataset({self.var_name: xr.Variable(dims, np.empty(shape, dtype=np.dtype('V1')), encoding={'preferred_chunks': dict(zip(dims, pref_chunks))})})

    def check_dataset(self, initial, final, expected_chunks):
        assert_identical(initial, final)
        assert final[self.var_name].chunks == expected_chunks

    @pytest.mark.parametrize('shape,pref_chunks', [((5,), (2,)), ((5,), ((2, 2, 1),)), ((5, 6), (4, 2)), ((5, 6), (4, (2, 2, 2)))])
    @pytest.mark.parametrize('request_with_empty_map', [False, True])
    def test_honor_chunks(self, shape, pref_chunks, request_with_empty_map):
        """Honor the backend's preferred chunks when opening a dataset."""
        initial = self.create_dataset(shape, pref_chunks)
        chunks = {} if request_with_empty_map else dict.fromkeys(initial[self.var_name].dims, None)
        final = xr.open_dataset(initial, engine=PassThroughBackendEntrypoint, chunks=chunks)
        self.check_dataset(initial, final, explicit_chunks(pref_chunks, shape))

    @pytest.mark.parametrize('shape,pref_chunks,req_chunks', [((5,), (2,), (3,)), ((5,), (2,), ((2, 1, 1, 1),)), ((5,), ((2, 2, 1),), (3,)), ((5,), ((2, 2, 1),), ((2, 1, 1, 1),)), ((1, 5), (1, 2), (1, 3))])
    def test_split_chunks(self, shape, pref_chunks, req_chunks):
        """Warn when the requested chunks separate the backend's preferred chunks."""
        initial = self.create_dataset(shape, pref_chunks)
        with pytest.warns(UserWarning):
            final = xr.open_dataset(initial, engine=PassThroughBackendEntrypoint, chunks=dict(zip(initial[self.var_name].dims, req_chunks)))
        self.check_dataset(initial, final, explicit_chunks(req_chunks, shape))

    @pytest.mark.parametrize('shape,pref_chunks,req_chunks', [((5,), (2,), (2,)), ((5,), (2,), ((2, 2, 1),)), ((5,), (2,), (4,)), ((5,), (2,), (6,)), ((5,), (1,), ((1, 1, 2, 1),)), ((5,), ((1, 1, 2, 1),), (2,)), ((5,), ((1, 1, 2, 1),), ((2, 3),)), ((5, 5), (2, (1, 1, 2, 1)), (4, (2, 3)))])
    def test_join_chunks(self, shape, pref_chunks, req_chunks):
        """Don't warn when the requested chunks join or keep the preferred chunks."""
        initial = self.create_dataset(shape, pref_chunks)
        with assert_no_warnings():
            final = xr.open_dataset(initial, engine=PassThroughBackendEntrypoint, chunks=dict(zip(initial[self.var_name].dims, req_chunks)))
        self.check_dataset(initial, final, explicit_chunks(req_chunks, shape))