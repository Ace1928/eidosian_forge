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
def _combine_cuda(aggs):
    ret = aggs[0]
    if len(aggs) > 1:
        kernel_args = cuda_args(ret.shape[:-1])
        if ret.ndim == 3:
            cuda_row_min_n_in_place_3d[kernel_args](aggs[0], aggs[1])
        else:
            cuda_row_min_n_in_place_4d[kernel_args](aggs[0], aggs[1])
    return ret