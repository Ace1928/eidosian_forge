from __future__ import annotations
from numbers import Number
from math import log10
import warnings
import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask.array as da
from xarray import DataArray, Dataset
from .utils import Dispatcher, ngjit, calc_res, calc_bbox, orient_array, \
from .utils import get_indices, dshape_from_pandas, dshape_from_dask
from .utils import Expr # noqa (API import)
from .resampling import resample_2d, resample_2d_distributed
from . import reductions as rd
def _broadcast_column_specifications(*args):
    lengths = {len(a) for a in args if isinstance(a, (list, tuple))}
    if len(lengths) != 1:
        return args
    else:
        n = lengths.pop()
        return tuple(((arg,) * n if isinstance(arg, (Number, str)) else arg for arg in args))