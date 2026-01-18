from __future__ import annotations
import os
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from functools import partial
from io import BytesIO
from numbers import Number
from typing import (
import numpy as np
from xarray import backends, conventions
from xarray.backends import plugins
from xarray.backends.common import (
from xarray.backends.locks import _get_scheduler
from xarray.backends.zarr import open_zarr
from xarray.core import indexing
from xarray.core.combine import (
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset, _get_chunk, _maybe_chunk
from xarray.core.indexes import Index
from xarray.core.types import ZarrWriteModes
from xarray.core.utils import is_remote_uri
from xarray.namedarray.daskmanager import DaskManager
from xarray.namedarray.parallelcompat import guess_chunkmanager
def _dataset_from_backend_dataset(backend_ds, filename_or_obj, engine, chunks, cache, overwrite_encoded_chunks, inline_array, chunked_array_type, from_array_kwargs, **extra_tokens):
    if not isinstance(chunks, (int, dict)) and chunks not in {None, 'auto'}:
        raise ValueError(f"chunks must be an int, dict, 'auto', or None. Instead found {chunks}.")
    _protect_dataset_variables_inplace(backend_ds, cache)
    if chunks is None:
        ds = backend_ds
    else:
        ds = _chunk_ds(backend_ds, filename_or_obj, engine, chunks, overwrite_encoded_chunks, inline_array, chunked_array_type, from_array_kwargs, **extra_tokens)
    ds.set_close(backend_ds._close)
    if 'source' not in ds.encoding and isinstance(filename_or_obj, (str, os.PathLike)):
        ds.encoding['source'] = _normalize_path(filename_or_obj)
    return ds