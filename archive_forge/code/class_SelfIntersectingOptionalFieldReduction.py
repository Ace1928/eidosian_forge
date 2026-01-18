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
class SelfIntersectingOptionalFieldReduction(OptionalFieldReduction):
    """
    Base class for optional field reductions for which self-intersecting
    geometry may or may not be desirable.
    Ignored if not using antialiasing.
    """

    def __init__(self, column=None, self_intersect=True):
        super().__init__(column)
        self.self_intersect = self_intersect

    def _antialias_requires_2_stages(self):
        return not self.self_intersect

    def _build_append(self, dshape, schema, cuda, antialias, self_intersect):
        if antialias and (not self_intersect):
            if cuda:
                if self.column is None:
                    return self._append_no_field_antialias_cuda_not_self_intersect
                else:
                    return self._append_antialias_cuda_not_self_intersect
            elif self.column is None:
                return self._append_no_field_antialias_not_self_intersect
            else:
                return self._append_antialias_not_self_intersect
        return super()._build_append(dshape, schema, cuda, antialias, self_intersect)

    def _hashable_inputs(self):
        return super()._hashable_inputs() + (self.self_intersect,)