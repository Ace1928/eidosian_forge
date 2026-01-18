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
def _validate_attrs(dataset, invalid_netcdf=False):
    """`attrs` must have a string key and a value which is either: a number,
    a string, an ndarray, a list/tuple of numbers/strings, or a numpy.bool_.

    Notes
    -----
    A numpy.bool_ is only allowed when using the h5netcdf engine with
    `invalid_netcdf=True`.
    """
    valid_types = (str, Number, np.ndarray, np.number, list, tuple)
    if invalid_netcdf:
        valid_types += (np.bool_,)

    def check_attr(name, value, valid_types):
        if isinstance(name, str):
            if not name:
                raise ValueError(f'Invalid name for attr {name!r}: string must be length 1 or greater for serialization to netCDF files')
        else:
            raise TypeError(f'Invalid name for attr: {name!r} must be a string for serialization to netCDF files')
        if not isinstance(value, valid_types):
            raise TypeError(f'Invalid value for attr {name!r}: {value!r}. For serialization to netCDF files, its value must be of one of the following types: {', '.join([vtype.__name__ for vtype in valid_types])}')
    for k, v in dataset.attrs.items():
        check_attr(k, v, valid_types)
    for variable in dataset.variables.values():
        for k, v in variable.attrs.items():
            check_attr(k, v, valid_types)