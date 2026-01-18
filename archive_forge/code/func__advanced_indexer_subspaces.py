from __future__ import annotations
import warnings
from typing import Callable
import numpy as np
import pandas as pd
from packaging.version import Version
from xarray.core.utils import is_duck_array, module_available
from xarray.namedarray import pycompat
from xarray.core.options import OPTIONS
def _advanced_indexer_subspaces(key):
    """Indices of the advanced indexes subspaces for mixed indexing and vindex."""
    if not isinstance(key, tuple):
        key = (key,)
    advanced_index_positions = [i for i, k in enumerate(key) if not isinstance(k, slice)]
    if not advanced_index_positions or not _is_contiguous(advanced_index_positions):
        return ((), ())
    non_slices = [k for k in key if not isinstance(k, slice)]
    broadcasted_shape = np.broadcast_shapes(*[item.shape if is_duck_array(item) else (0,) for item in non_slices])
    ndim = len(broadcasted_shape)
    mixed_positions = advanced_index_positions[0] + np.arange(ndim)
    vindex_positions = np.arange(ndim)
    return (mixed_positions, vindex_positions)