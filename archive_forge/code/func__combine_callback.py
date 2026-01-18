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
def _combine_callback(self, cuda, partitioned, categorical):
    selector = self.selector
    is_n_reduction = isinstance(selector, FloatingNReduction)
    if cuda:
        append = selector._append_cuda
    else:
        append = selector._append
    invalid = isminus1 if self.selector.uses_row_index(cuda, partitioned) else isnull

    @ngjit
    def combine_cpu_2d(aggs, selector_aggs):
        ny, nx = aggs[0].shape
        for y in range(ny):
            for x in range(nx):
                value = selector_aggs[1][y, x]
                if not invalid(value) and append(x, y, selector_aggs[0], value) >= 0:
                    aggs[0][y, x] = aggs[1][y, x]

    @ngjit
    def combine_cpu_3d(aggs, selector_aggs):
        ny, nx, ncat = aggs[0].shape
        for y in range(ny):
            for x in range(nx):
                for cat in range(ncat):
                    value = selector_aggs[1][y, x, cat]
                    if not invalid(value) and append(x, y, selector_aggs[0][:, :, cat], value) >= 0:
                        aggs[0][y, x, cat] = aggs[1][y, x, cat]

    @ngjit
    def combine_cpu_n_3d(aggs, selector_aggs):
        ny, nx, n = aggs[0].shape
        for y in range(ny):
            for x in range(nx):
                for i in range(n):
                    value = selector_aggs[1][y, x, i]
                    if invalid(value):
                        break
                    update_index = append(x, y, selector_aggs[0], value)
                    if update_index < 0:
                        break
                    shift_and_insert(aggs[0][y, x], aggs[1][y, x, i], update_index)

    @ngjit
    def combine_cpu_n_4d(aggs, selector_aggs):
        ny, nx, ncat, n = aggs[0].shape
        for y in range(ny):
            for x in range(nx):
                for cat in range(ncat):
                    for i in range(n):
                        value = selector_aggs[1][y, x, cat, i]
                        if invalid(value):
                            break
                        update_index = append(x, y, selector_aggs[0][:, :, cat, :], value)
                        if update_index < 0:
                            break
                        shift_and_insert(aggs[0][y, x, cat], aggs[1][y, x, cat, i], update_index)

    @nb_cuda.jit
    def combine_cuda_2d(aggs, selector_aggs):
        ny, nx = aggs[0].shape
        x, y = nb_cuda.grid(2)
        if x < nx and y < ny:
            value = selector_aggs[1][y, x]
            if not invalid(value) and append(x, y, selector_aggs[0], value) >= 0:
                aggs[0][y, x] = aggs[1][y, x]

    @nb_cuda.jit
    def combine_cuda_3d(aggs, selector_aggs):
        ny, nx, ncat = aggs[0].shape
        x, y, cat = nb_cuda.grid(3)
        if x < nx and y < ny and (cat < ncat):
            value = selector_aggs[1][y, x, cat]
            if not invalid(value) and append(x, y, selector_aggs[0][:, :, cat], value) >= 0:
                aggs[0][y, x, cat] = aggs[1][y, x, cat]

    @nb_cuda.jit
    def combine_cuda_n_3d(aggs, selector_aggs):
        ny, nx, n = aggs[0].shape
        x, y = nb_cuda.grid(2)
        if x < nx and y < ny:
            for i in range(n):
                value = selector_aggs[1][y, x, i]
                if invalid(value):
                    break
                update_index = append(x, y, selector_aggs[0], value)
                if update_index < 0:
                    break
                cuda_shift_and_insert(aggs[0][y, x], aggs[1][y, x, i], update_index)

    @nb_cuda.jit
    def combine_cuda_n_4d(aggs, selector_aggs):
        ny, nx, ncat, n = aggs[0].shape
        x, y, cat = nb_cuda.grid(3)
        if x < nx and y < ny and (cat < ncat):
            for i in range(n):
                value = selector_aggs[1][y, x, cat, i]
                if invalid(value):
                    break
                update_index = append(x, y, selector_aggs[0][:, :, cat, :], value)
                if update_index < 0:
                    break
                cuda_shift_and_insert(aggs[0][y, x, cat], aggs[1][y, x, cat, i], update_index)
    if is_n_reduction:
        if cuda:
            return combine_cuda_n_4d if categorical else combine_cuda_n_3d
        else:
            return combine_cpu_n_4d if categorical else combine_cpu_n_3d
    elif cuda:
        return combine_cuda_3d if categorical else combine_cuda_2d
    else:
        return combine_cpu_3d if categorical else combine_cpu_2d