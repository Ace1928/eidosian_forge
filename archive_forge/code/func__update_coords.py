from __future__ import annotations
from collections.abc import Hashable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import (
import numpy as np
import pandas as pd
from xarray.core import formatting
from xarray.core.alignment import Aligner
from xarray.core.indexes import (
from xarray.core.merge import merge_coordinates_without_align, merge_coords
from xarray.core.types import DataVars, Self, T_DataArray, T_Xarray
from xarray.core.utils import (
from xarray.core.variable import Variable, as_variable, calculate_dimensions
def _update_coords(self, coords: dict[Hashable, Variable], indexes: Mapping[Any, Index]) -> None:
    coords_plus_data = coords.copy()
    coords_plus_data[_THIS_ARRAY] = self._data.variable
    dims = calculate_dimensions(coords_plus_data)
    if not set(dims) <= set(self.dims):
        raise ValueError('cannot add coordinates with new dimensions to a DataArray')
    self._data._coords = coords
    original_indexes = dict(self._data.xindexes)
    original_indexes.update(indexes)
    self._data._indexes = original_indexes