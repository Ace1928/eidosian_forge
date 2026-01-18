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
class _count_ignore_antialiasing(count):
    """Count reduction but ignores antialiasing. Used by mean reduction.
    """

    def out_dshape(self, in_dshape, antialias, cuda, partitioned):
        return dshape(ct.uint32)

    def _antialias_stage_2(self, self_intersect, array_module) -> tuple[AntialiasStage2]:
        if self_intersect:
            return (AntialiasStage2(AntialiasCombination.SUM_1AGG, 0),)
        else:
            return (AntialiasStage2(AntialiasCombination.SUM_2AGG, 0),)

    @staticmethod
    @ngjit
    def _append_antialias(x, y, agg, field, aa_factor, prev_aa_factor):
        if not isnull(field) and prev_aa_factor == 0.0:
            agg[y, x] += 1
            return 0
        return -1

    @staticmethod
    @ngjit
    def _append_antialias_not_self_intersect(x, y, agg, field, aa_factor, prev_aa_factor):
        if not isnull(field) and prev_aa_factor == 0.0:
            agg[y, x] += 1
            return 0
        return -1