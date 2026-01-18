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
@contextmanager
def _merge_inplace(self, other):
    """For use with in-place binary arithmetic."""
    if other is None:
        yield
    else:
        prioritized = {k: (v, None) for k, v in self.variables.items() if k not in self.xindexes}
        variables, indexes = merge_coordinates_without_align([self, other], prioritized)
        yield
        self._update_coords(variables, indexes)