from __future__ import annotations
import itertools
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Literal, Union
import pandas as pd
from xarray.core import dtypes
from xarray.core.concat import concat
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.merge import merge
from xarray.core.utils import iterate_nested
def _check_shape_tile_ids(combined_tile_ids):
    """Check all lists along one dimension are same length."""
    tile_ids, nesting_depths = _check_dimension_depth_tile_ids(combined_tile_ids)
    for dim in range(nesting_depths[0]):
        indices_along_dim = [tile_id[dim] for tile_id in tile_ids]
        occurrences = Counter(indices_along_dim)
        if len(set(occurrences.values())) != 1:
            raise ValueError(f'The supplied objects do not form a hypercube because sub-lists do not have consistent lengths along dimension {dim}')