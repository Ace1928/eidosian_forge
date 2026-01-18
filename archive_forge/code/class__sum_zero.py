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
class _sum_zero(FloatingReduction):
    """Sum of all elements in ``column``.

    Parameters
    ----------
    column : str
        Name of the column to aggregate over. Column data type must be numeric.
    """

    def _antialias_stage_2(self, self_intersect, array_module) -> tuple[AntialiasStage2]:
        if self_intersect:
            return (AntialiasStage2(AntialiasCombination.SUM_1AGG, 0),)
        else:
            return (AntialiasStage2(AntialiasCombination.SUM_2AGG, 0),)

    def _build_create(self, required_dshape):
        return self._create_float64_zero

    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        if not isnull(field):
            agg[y, x] += field
            return 0
        return -1

    @staticmethod
    @ngjit
    def _append_antialias(x, y, agg, field, aa_factor, prev_aa_factor):
        value = field * (aa_factor - prev_aa_factor)
        if not isnull(value):
            agg[y, x] += value
            return 0
        return -1

    @staticmethod
    @ngjit
    def _append_antialias_not_self_intersect(x, y, agg, field, aa_factor, prev_aa_factor):
        value = field * aa_factor
        if not isnull(value) and value > agg[y, x]:
            agg[y, x] = value
            return 0
        return -1

    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_cuda(x, y, agg, field):
        if not isnull(field):
            nb_cuda.atomic.add(agg, (y, x), field)
            return 0
        return -1

    @staticmethod
    def _combine(aggs):
        return aggs.sum(axis=0, dtype='f8')