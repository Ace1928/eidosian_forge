from warnings import warn
import warnings
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from shapely.geometry import box, MultiPoint
from shapely.geometry.base import BaseGeometry
from . import _compat as compat
from .array import GeometryArray, GeometryDtype, points_from_xy
def _get_index_for_parts(orig_idx, outer_idx, ignore_index, index_parts):
    """Helper to handle index when geometries get exploded to parts.

    Used in get_coordinates and explode.

    Parameters
    ----------
    orig_idx : pandas.Index
        original index
    outer_idx : array
        the index of each returned geometry as a separate ndarray of integers
    ignore_index : bool
    index_parts : bool

    Returns
    -------
    pandas.Index
        index or multiindex
    """
    if ignore_index:
        return None
    else:
        if len(outer_idx):
            run_start = np.r_[True, outer_idx[:-1] != outer_idx[1:]]
            counts = np.diff(np.r_[np.nonzero(run_start)[0], len(outer_idx)])
            inner_index = (~run_start).cumsum(dtype=outer_idx.dtype)
            inner_index -= np.repeat(inner_index[run_start], counts)
        else:
            inner_index = []
        outer_index = orig_idx.take(outer_idx)
        if index_parts:
            nlevels = outer_index.nlevels
            index_arrays = [outer_index.get_level_values(lvl) for lvl in range(nlevels)]
            index_arrays.append(inner_index)
            index = pd.MultiIndex.from_arrays(index_arrays, names=orig_idx.names + [None])
        else:
            index = outer_index
    return index