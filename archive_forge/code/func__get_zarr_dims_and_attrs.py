from __future__ import annotations
import json
import os
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any
import numpy as np
from xarray import coding, conventions
from xarray.backends.common import (
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.types import ZarrWriteModes
from xarray.core.utils import (
from xarray.core.variable import Variable
from xarray.namedarray.parallelcompat import guess_chunkmanager
from xarray.namedarray.pycompat import integer_types
def _get_zarr_dims_and_attrs(zarr_obj, dimension_key, try_nczarr):
    try:
        dimensions = zarr_obj.attrs[dimension_key]
    except KeyError as e:
        if not try_nczarr:
            raise KeyError(f'Zarr object is missing the attribute `{dimension_key}`, which is required for xarray to determine variable dimensions.') from e
        zarray_path = os.path.join(zarr_obj.path, '.zarray')
        zarray = json.loads(zarr_obj.store[zarray_path])
        try:
            dimensions = [os.path.basename(dim) for dim in zarray['_NCZARR_ARRAY']['dimrefs']]
        except KeyError as e:
            raise KeyError(f'Zarr object is missing the attribute `{dimension_key}` and the NCZarr metadata, which are required for xarray to determine variable dimensions.') from e
    nc_attrs = [attr for attr in zarr_obj.attrs if attr.lower().startswith('_nc')]
    attributes = HiddenKeyDict(zarr_obj.attrs, [dimension_key] + nc_attrs)
    return (dimensions, attributes)