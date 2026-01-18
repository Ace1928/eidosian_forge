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
def _build_finalize(self, dshape):
    if isinstance(self.selector, FloatingNReduction):
        add_finalize_kwargs = self.selector._add_finalize_kwargs
    else:
        add_finalize_kwargs = None

    def finalize(bases, cuda=False, **kwargs):
        if add_finalize_kwargs is not None:
            kwargs = add_finalize_kwargs(**kwargs)
        return xr.DataArray(bases[-1], **kwargs)
    return finalize