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
class category_binning(category_modulo):
    """
    A variation on category_codes that assigns categories by binning a continuous-valued column.
    The number of categories returned is always nbins+1.
    The last category (nbin) is for NaNs in the data column, as well as for values under/over the
    binned interval (when include_under or include_over is False).

    Parameters
    ----------
    column:   column to use
    lower:    lower bound of first bin
    upper:    upper bound of last bin
    nbins:     number of bins
    include_under: if True, values below bin 0 are assigned to category 0
    include_over:  if True, values above the last bin (nbins-1) are assigned to category nbin-1
    """

    def __init__(self, column, lower, upper, nbins, include_under=True, include_over=True):
        super().__init__(column, nbins + 1)
        self.bin0 = lower
        self.binsize = (upper - lower) / float(nbins)
        self.nbins = nbins
        self.bin_under = 0 if include_under else nbins
        self.bin_over = nbins - 1 if include_over else nbins

    def _hashable_inputs(self):
        return super()._hashable_inputs() + (self.bin0, self.binsize, self.bin_under, self.bin_over)

    def validate(self, in_dshape):
        if self.column not in in_dshape.dict:
            raise ValueError('specified column not found')

    def apply(self, df, cuda):
        if cudf and isinstance(df, cudf.DataFrame):
            if Version(cudf.__version__) >= Version('22.02'):
                values = df[self.column].to_cupy(na_value=cp.nan)
            else:
                values = cp.array(df[self.column].to_gpu_array(fillna=True))
            nan_values = cp.isnan(values)
        else:
            values = df[self.column].to_numpy()
            nan_values = np.isnan(values)
        index_float = (values - self.bin0) / self.binsize
        index_float[nan_values] = 0
        index = index_float.astype(int)
        index[index < 0] = self.bin_under
        index[index >= self.nbins] = self.bin_over
        index[nan_values] = self.nbins
        return index