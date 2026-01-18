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
@staticmethod
@nb_cuda.jit(device=True)
def _append_cuda(x, y, agg, field):
    if field != -1:
        n = agg.shape[2]
        for i in range(n):
            if agg[y, x, i] == -1 or field < agg[y, x, i]:
                cuda_shift_and_insert(agg[y, x], field, i)
                return i
    return -1