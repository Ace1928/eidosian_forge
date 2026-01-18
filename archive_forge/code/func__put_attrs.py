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
def _put_attrs(zarr_obj, attrs):
    """Raise a more informative error message for invalid attrs."""
    try:
        zarr_obj.attrs.put(attrs)
    except TypeError as e:
        raise TypeError('Invalid attribute in Dataset.attrs.') from e
    return zarr_obj