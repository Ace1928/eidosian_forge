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
class CategoryPreprocess(Preprocess):
    """Base class for categorizing preprocessors."""

    @property
    def cat_column(self):
        """Returns name of categorized column"""
        return self.column

    def categories(self, input_dshape):
        """Returns list of categories corresponding to input shape"""
        raise NotImplementedError('categories not implemented')

    def validate(self, in_dshape):
        """Validates input shape"""
        raise NotImplementedError('validate not implemented')

    def apply(self, df, cuda):
        """Applies preprocessor to DataFrame and returns array"""
        raise NotImplementedError('apply not implemented')