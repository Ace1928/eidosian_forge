from __future__ import annotations
import copy
from enum import Enum
from packaging.version import Version
import numpy as np
from datashader.datashape import dshape, isnumeric, Record, Option
from datashader.datashape import coretypes as ct
from toolz import concat, unique
import xarray as xr
from datashader.antialias import AntialiasCombination, AntialiasStage2
from datashader.utils import isminus1, isnull
from numba import cuda as nb_cuda
from .utils import (
class _max_or_min_row_index(OptionalFieldReduction):
    """Abstract base class of max and min row_index reductions.
    """

    def __init__(self):
        super().__init__(column=SpecialColumn.RowIndex)

    def out_dshape(self, in_dshape, antialias, cuda, partitioned):
        return dshape(ct.int64)

    def uses_row_index(self, cuda, partitioned):
        return True