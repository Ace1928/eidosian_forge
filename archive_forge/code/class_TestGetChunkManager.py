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
class TestGetChunkManager:

    def test_get_chunkmanger(self, register_dummy_chunkmanager) -> None:
        chunkmanager = guess_chunkmanager('dummy')
        assert isinstance(chunkmanager, DummyChunkManager)

    def test_fail_on_nonexistent_chunkmanager(self) -> None:
        with pytest.raises(ValueError, match='unrecognized chunk manager foo'):
            guess_chunkmanager('foo')

    @requires_dask
    def test_get_dask_if_installed(self) -> None:
        chunkmanager = guess_chunkmanager(None)
        assert isinstance(chunkmanager, DaskManager)

    @pytest.mark.skipif(has_dask, reason='requires dask not to be installed')
    def test_dont_get_dask_if_not_installed(self) -> None:
        with pytest.raises(ValueError, match='unrecognized chunk manager dask'):
            guess_chunkmanager('dask')

    @requires_dask
    def test_choose_dask_over_other_chunkmanagers(self, register_dummy_chunkmanager) -> None:
        chunk_manager = guess_chunkmanager(None)
        assert isinstance(chunk_manager, DaskManager)