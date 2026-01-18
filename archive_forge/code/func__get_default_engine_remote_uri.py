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
def _get_default_engine_remote_uri() -> Literal['netcdf4', 'pydap']:
    engine: Literal['netcdf4', 'pydap']
    try:
        import netCDF4
        engine = 'netcdf4'
    except ImportError:
        try:
            import pydap
            engine = 'pydap'
        except ImportError:
            raise ValueError('netCDF4 or pydap is required for accessing remote datasets via OPeNDAP')
    return engine