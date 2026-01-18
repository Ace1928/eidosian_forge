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
def _antialias_stage_2(self, self_intersect, array_module) -> tuple[AntialiasStage2]:
    return (AntialiasStage2(AntialiasCombination.MIN, -1, n_reduction=True),)