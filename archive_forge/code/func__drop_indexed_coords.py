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
def _drop_indexed_coords(self, coords_to_drop: set[Hashable]) -> None:
    assert self._data.xindexes is not None
    new_coords = drop_indexed_coords(coords_to_drop, self)
    for name in self._data._coord_names - new_coords._names:
        del self._data._variables[name]
    self._data._indexes = dict(new_coords.xindexes)
    self._data._coord_names.intersection_update(new_coords._names)