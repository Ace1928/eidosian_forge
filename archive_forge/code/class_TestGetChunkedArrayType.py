from __future__ import annotations
from importlib.metadata import EntryPoint
from typing import Any
import numpy as np
import pytest
from xarray.core.types import T_Chunks, T_DuckArray, T_NormalizedChunks
from xarray.namedarray._typing import _Chunks
from xarray.namedarray.daskmanager import DaskManager
from xarray.namedarray.parallelcompat import (
from xarray.tests import has_dask, requires_dask
class TestGetChunkedArrayType:

    def test_detect_chunked_arrays(self, register_dummy_chunkmanager) -> None:
        dummy_arr = DummyChunkedArray([1, 2, 3])
        chunk_manager = get_chunked_array_type(dummy_arr)
        assert isinstance(chunk_manager, DummyChunkManager)

    def test_ignore_inmemory_arrays(self, register_dummy_chunkmanager) -> None:
        dummy_arr = DummyChunkedArray([1, 2, 3])
        chunk_manager = get_chunked_array_type(*[dummy_arr, 1.0, np.array([5, 6])])
        assert isinstance(chunk_manager, DummyChunkManager)
        with pytest.raises(TypeError, match='Expected a chunked array'):
            get_chunked_array_type(5.0)

    def test_raise_if_no_arrays_chunked(self, register_dummy_chunkmanager) -> None:
        with pytest.raises(TypeError, match='Expected a chunked array '):
            get_chunked_array_type(*[1.0, np.array([5, 6])])

    def test_raise_if_no_matching_chunkmanagers(self) -> None:
        dummy_arr = DummyChunkedArray([1, 2, 3])
        with pytest.raises(TypeError, match='Could not find a Chunk Manager which recognises'):
            get_chunked_array_type(dummy_arr)

    @requires_dask
    def test_detect_dask_if_installed(self) -> None:
        import dask.array as da
        dask_arr = da.from_array([1, 2, 3], chunks=(1,))
        chunk_manager = get_chunked_array_type(dask_arr)
        assert isinstance(chunk_manager, DaskManager)

    @requires_dask
    def test_raise_on_mixed_array_types(self, register_dummy_chunkmanager) -> None:
        import dask.array as da
        dummy_arr = DummyChunkedArray([1, 2, 3])
        dask_arr = da.from_array([1, 2, 3], chunks=(1,))
        with pytest.raises(TypeError, match='received multiple types'):
            get_chunked_array_type(*[dask_arr, dummy_arr])