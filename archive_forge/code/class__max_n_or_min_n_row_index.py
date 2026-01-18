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
class _max_n_or_min_n_row_index(FloatingNReduction):
    """Abstract base class of max_n and min_n row_index reductions.
    """

    def __init__(self, n=1):
        super().__init__(column=SpecialColumn.RowIndex)
        self.n = n if n >= 1 else 1

    def out_dshape(self, in_dshape, antialias, cuda, partitioned):
        return dshape(ct.int64)

    def uses_cuda_mutex(self) -> UsesCudaMutex:
        return UsesCudaMutex.Local

    def uses_row_index(self, cuda, partitioned):
        return True

    def _build_combine(self, dshape, antialias, cuda, partitioned, categorical=False):
        if cuda:
            return self._combine_cuda
        else:
            return self._combine