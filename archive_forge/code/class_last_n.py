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
class last_n(_first_n_or_last_n):

    def _antialias_stage_2(self, self_intersect, array_module) -> tuple[AntialiasStage2]:
        return (AntialiasStage2(AntialiasCombination.LAST, array_module.nan, n_reduction=True),)

    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        if not isnull(field):
            shift_and_insert(agg[y, x], field, 0)
            return 0
        return -1

    @staticmethod
    @ngjit
    def _append_antialias(x, y, agg, field, aa_factor, prev_aa_factor):
        value = field * aa_factor
        if not isnull(value):
            shift_and_insert(agg[y, x], value, 0)
            return 0
        return -1

    def _create_row_index_selector(self):
        return _max_n_row_index(n=self.n)