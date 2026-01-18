from __future__ import annotations
import collections
import itertools
import operator
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable, Literal, TypedDict
import numpy as np
from xarray.core.alignment import align
from xarray.core.coordinates import Coordinates
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.indexes import Index
from xarray.core.merge import merge
from xarray.core.utils import is_dask_collection
from xarray.core.variable import Variable
def _get_chunk_slicer(dim: Hashable, chunk_index: Mapping, chunk_bounds: Mapping):
    if dim in chunk_index:
        which_chunk = chunk_index[dim]
        return slice(chunk_bounds[dim][which_chunk], chunk_bounds[dim][which_chunk + 1])
    return slice(None)