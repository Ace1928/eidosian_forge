from __future__ import annotations
import dask.array as da
from dask import config
from dask.array.backends import ArrayBackendEntrypoint, register_cupy
from dask.array.core import Array
from dask.array.dispatch import to_cupy_dispatch
def _da_with_cupy_meta(attr, *args, meta=None, **kwargs):
    meta = _cupy().empty(()) if meta is None else meta
    with config.set({'array.backend': 'numpy'}):
        return getattr(da, attr)(*args, meta=meta, **kwargs)