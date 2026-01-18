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
class category_codes(CategoryPreprocess):
    """
    Extract just the category codes from a categorical column.

    To create a new type of categorizer, derive a subclass from this
    class or one of its subclasses, implementing ``__init__``,
    ``_hashable_inputs``, ``categories``, ``validate``, and ``apply``.

    See the implementation of ``category_modulo`` in ``reductions.py``
    for an example.
    """

    def categories(self, input_dshape):
        return input_dshape.measure[self.column].categories

    def validate(self, in_dshape):
        if self.column not in in_dshape.dict:
            raise ValueError('specified column not found')
        if not isinstance(in_dshape.measure[self.column], ct.Categorical):
            raise ValueError('input must be categorical')

    def apply(self, df, cuda):
        if cudf and isinstance(df, cudf.DataFrame):
            if Version(cudf.__version__) >= Version('22.02'):
                return df[self.column].cat.codes.to_cupy()
            return df[self.column].cat.codes.to_gpu_array()
        else:
            return df[self.column].cat.codes.values