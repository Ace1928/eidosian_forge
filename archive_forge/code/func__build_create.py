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
def _build_create(self, required_dshape):
    if isinstance(self.selector, FloatingNReduction):
        return lambda shape, array_module: super(where, self)._build_create(required_dshape)(shape + (self.selector.n,), array_module)
    else:
        return super()._build_create(required_dshape)