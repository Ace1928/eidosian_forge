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
class max_n(FloatingNReduction):

    def uses_cuda_mutex(self) -> UsesCudaMutex:
        return UsesCudaMutex.Local

    def _antialias_stage_2(self, self_intersect, array_module) -> tuple[AntialiasStage2]:
        return (AntialiasStage2(AntialiasCombination.MAX, array_module.nan, n_reduction=True),)

    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        if not isnull(field):
            n = agg.shape[2]
            for i in range(n):
                if isnull(agg[y, x, i]) or field > agg[y, x, i]:
                    shift_and_insert(agg[y, x], field, i)
                    return i
        return -1

    @staticmethod
    @ngjit
    def _append_antialias(x, y, agg, field, aa_factor, prev_aa_factor):
        value = field * aa_factor
        if not isnull(value):
            n = agg.shape[2]
            for i in range(n):
                if isnull(agg[y, x, i]) or value > agg[y, x, i]:
                    shift_and_insert(agg[y, x], value, i)
                    return i
        return -1

    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_cuda(x, y, agg, field):
        if not isnull(field):
            n = agg.shape[2]
            for i in range(n):
                if isnull(agg[y, x, i]) or field > agg[y, x, i]:
                    cuda_shift_and_insert(agg[y, x], field, i)
                    return i
        return -1

    def _build_combine(self, dshape, antialias, cuda, partitioned, categorical=False):
        if cuda:
            return self._combine_cuda
        else:
            return self._combine

    @staticmethod
    def _combine(aggs):
        ret = aggs[0]
        for i in range(1, len(aggs)):
            if ret.ndim == 3:
                nanmax_n_in_place_3d(aggs[0], aggs[i])
            else:
                nanmax_n_in_place_4d(aggs[0], aggs[i])
        return ret

    @staticmethod
    def _combine_cuda(aggs):
        ret = aggs[0]
        kernel_args = cuda_args(ret.shape[:-1])
        for i in range(1, len(aggs)):
            if ret.ndim == 3:
                cuda_nanmax_n_in_place_3d[kernel_args](aggs[0], aggs[i])
            else:
                cuda_nanmax_n_in_place_4d[kernel_args](aggs[0], aggs[i])
        return ret