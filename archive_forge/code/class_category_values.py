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
class category_values(CategoryPreprocess):
    """Extract a category and a value column from a dataframe as (2,N) numpy array of values."""

    def __init__(self, categorizer, value_column):
        super().__init__(value_column)
        self.categorizer = categorizer

    @property
    def inputs(self):
        return (self.categorizer.column, self.column)

    @property
    def cat_column(self):
        """Returns name of categorized column"""
        return self.categorizer.column

    def categories(self, input_dshape):
        return self.categorizer.categories

    def validate(self, in_dshape):
        return self.categorizer.validate(in_dshape)

    def apply(self, df, cuda):
        a = self.categorizer.apply(df, cuda)
        if cudf and isinstance(df, cudf.DataFrame):
            import cupy
            if self.column == SpecialColumn.RowIndex:
                nullval = -1
            elif df[self.column].dtype.kind == 'f':
                nullval = np.nan
            else:
                nullval = 0
            a = cupy.asarray(a)
            if self.column == SpecialColumn.RowIndex:
                b = extract(SpecialColumn.RowIndex).apply(df, cuda)
            elif Version(cudf.__version__) >= Version('22.02'):
                b = df[self.column].to_cupy(na_value=nullval)
            else:
                b = cupy.asarray(df[self.column].fillna(nullval))
            return cupy.stack((a, b), axis=-1)
        else:
            if self.column == SpecialColumn.RowIndex:
                b = extract(SpecialColumn.RowIndex).apply(df, cuda)
            else:
                b = df[self.column].values
            return np.stack((a, b), axis=-1)