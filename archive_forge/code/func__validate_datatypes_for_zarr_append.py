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
def _validate_datatypes_for_zarr_append(zstore, dataset):
    """If variable exists in the store, confirm dtype of the data to append is compatible with
    existing dtype.
    """
    existing_vars = zstore.get_variables()

    def check_dtype(vname, var):
        if vname not in existing_vars or np.issubdtype(var.dtype, np.number) or np.issubdtype(var.dtype, np.datetime64) or np.issubdtype(var.dtype, np.bool_) or (var.dtype == object):
            pass
        elif not var.dtype == existing_vars[vname].dtype:
            raise ValueError(f'Mismatched dtypes for variable {vname} between Zarr store on disk and dataset to append. Store has dtype {existing_vars[vname].dtype} but dataset to append has dtype {var.dtype}.')
    for vname, var in dataset.data_vars.items():
        check_dtype(vname, var)