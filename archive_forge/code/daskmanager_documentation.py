from __future__ import annotations
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Callable
import numpy as np
from packaging.version import Version
from xarray.core.indexing import ImplicitToExplicitIndexingAdapter
from xarray.namedarray.parallelcompat import ChunkManagerEntrypoint, T_ChunkedArray
from xarray.namedarray.utils import is_duck_dask_array, module_available
Called by open_dataset