from __future__ import annotations
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any
import numpy as np
from xarray.backends.common import (
from xarray.backends.file_manager import CachingFileManager
from xarray.backends.locks import (
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.utils import Frozen, FrozenDict, close_on_error
from xarray.core.variable import Variable
class PynioBackendEntrypoint(BackendEntrypoint):
    """
    PyNIO backend

        .. deprecated:: 0.20.0

        Deprecated as PyNIO is no longer supported. See
        https://github.com/pydata/xarray/issues/4491 for more information
    """

    def open_dataset(self, filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore, *, mask_and_scale=True, decode_times=True, concat_characters=True, decode_coords=True, drop_variables: str | Iterable[str] | None=None, use_cftime=None, decode_timedelta=None, mode='r', lock=None) -> Dataset:
        filename_or_obj = _normalize_path(filename_or_obj)
        store = NioDataStore(filename_or_obj, mode=mode, lock=lock)
        store_entrypoint = StoreBackendEntrypoint()
        with close_on_error(store):
            ds = store_entrypoint.open_dataset(store, mask_and_scale=mask_and_scale, decode_times=decode_times, concat_characters=concat_characters, decode_coords=decode_coords, drop_variables=drop_variables, use_cftime=use_cftime, decode_timedelta=decode_timedelta)
        return ds