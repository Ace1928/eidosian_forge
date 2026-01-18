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
class count_cat(by):
    """Count of all elements in ``column``, grouped by category.
    Alias for `by(...,count())`, for backwards compatibility.

    Parameters
    ----------
    column : str
        Name of the column to aggregate over. Column data type must be
        categorical. Resulting aggregate has a outer dimension axis along the
        categories present.
    """

    def __init__(self, column):
        super(count_cat, self).__init__(column, count())