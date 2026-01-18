from __future__ import annotations
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.missing import NA
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.internals.array_manager import ArrayManager
from pandas.core.internals.blocks import (
from pandas.core.internals.managers import (
def _concat_homogeneous_fastpath(mgrs_indexers, shape: Shape, first_dtype: np.dtype) -> Block:
    """
    With single-Block managers with homogeneous dtypes (that can already hold nan),
    we avoid [...]
    """
    if all((not indexers for _, indexers in mgrs_indexers)):
        arrs = [mgr.blocks[0].values.T for mgr, _ in mgrs_indexers]
        arr = np.concatenate(arrs).T
        bp = libinternals.BlockPlacement(slice(shape[0]))
        nb = new_block_2d(arr, bp)
        return nb
    arr = np.empty(shape, dtype=first_dtype)
    if first_dtype == np.float64:
        take_func = libalgos.take_2d_axis0_float64_float64
    else:
        take_func = libalgos.take_2d_axis0_float32_float32
    start = 0
    for mgr, indexers in mgrs_indexers:
        mgr_len = mgr.shape[1]
        end = start + mgr_len
        if 0 in indexers:
            take_func(mgr.blocks[0].values, indexers[0], arr[:, start:end])
        else:
            arr[:, start:end] = mgr.blocks[0].values
        start += mgr_len
    bp = libinternals.BlockPlacement(slice(shape[0]))
    nb = new_block_2d(arr, bp)
    return nb