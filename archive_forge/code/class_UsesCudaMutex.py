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
class UsesCudaMutex(Enum):
    """
    Enum that encapsulates the need for a Reduction to use a CUDA mutex to
    operate correctly on a GPU. Possible values:

    No: the Reduction append_cuda function is atomic and no mutex is required.
    Local: Reduction append_cuda needs wrapping in a mutex.
    Global: the overall compiled append function needs wrapping in a mutex.
    """
    No = 0
    Local = 1
    Global = 2