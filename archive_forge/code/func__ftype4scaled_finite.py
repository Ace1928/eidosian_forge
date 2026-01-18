from __future__ import annotations
import io
import sys
import typing as ty
import warnings
from functools import reduce
from operator import getitem, mul
from os.path import exists, splitext
import numpy as np
from ._compression import COMPRESSED_FILE_LIKES
from .casting import OK_FLOATS, shared_range
from .externals.oset import OrderedSet
def _ftype4scaled_finite(tst_arr: np.ndarray, slope: npt.ArrayLike, inter: npt.ArrayLike, direction: ty.Literal['read', 'write']='read', default: type[np.floating]=np.float32) -> type[np.floating]:
    """Smallest float type for scaling of `tst_arr` that does not overflow"""
    assert direction in ('read', 'write')
    if default not in OK_FLOATS and default is np.longdouble:
        return default
    def_ind = OK_FLOATS.index(default)
    tst_arr = np.atleast_1d(tst_arr)
    slope = np.atleast_1d(slope)
    inter = np.atleast_1d(inter)
    for ftype in OK_FLOATS[def_ind:]:
        tst_trans = tst_arr.copy()
        slope = slope.astype(ftype)
        inter = inter.astype(ftype)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('error', '.*overflow.*', RuntimeWarning)
                if direction == 'read':
                    if slope != 1.0:
                        tst_trans = tst_trans * slope
                    if inter != 0.0:
                        tst_trans = tst_trans + inter
                elif direction == 'write':
                    if inter != 0.0:
                        tst_trans = tst_trans - inter
                    if slope != 1.0:
                        tst_trans = tst_trans / slope
            if np.all(np.isfinite(tst_trans)):
                return ftype
        except RuntimeWarning:
            pass
    raise ValueError('Overflow using highest floating point type')