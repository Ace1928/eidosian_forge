from __future__ import annotations
from itertools import groupby
from math import floor, ceil
import dask.array as da
import numpy as np
from dask.delayed import delayed
from numba import prange
from .utils import ngjit, ngjit_parallel
def _resample_2d(src, mask, use_mask, ds_method, us_method, fill_value, mode_rank, x_offset, y_offset, out):
    src_w, src_h, out_w, out_h = _get_dimensions(src, out)
    x0_off, x1_off = x_offset
    y0_off, y1_off = y_offset
    src_wo = src_w - x0_off - x1_off
    src_ho = src_h - y0_off - y1_off
    if us_method not in UPSAMPLING_METHODS:
        raise ValueError('invalid upsampling method')
    elif ds_method not in DOWNSAMPLING_METHODS:
        raise ValueError('invalid downsampling method')
    downsampling_method = DOWNSAMPLING_METHODS[ds_method]
    upsampling_method = UPSAMPLING_METHODS[us_method]
    if src_h == 0 or src_w == 0 or out_h == 0 or (out_w == 0):
        return np.zeros((out_h, out_w), dtype=src.dtype)
    elif out_w < src_wo and out_h < src_ho:
        return downsampling_method(src, mask, use_mask, ds_method, fill_value, mode_rank, x_offset, y_offset, out)
    elif out_w < src_wo:
        if out_h > src_ho:
            temp = np.zeros((src_h, out_w), dtype=src.dtype)
            temp = downsampling_method(src, mask, use_mask, ds_method, fill_value, mode_rank, x_offset, y_offset, temp)
            return upsampling_method(temp, mask, use_mask, fill_value, x_offset, y_offset, out)
        else:
            return downsampling_method(src, mask, use_mask, ds_method, fill_value, mode_rank, x_offset, y_offset, out)
    elif out_h < src_ho:
        if out_w > src_wo:
            temp = np.zeros((out_h, src_w), dtype=src.dtype)
            temp = downsampling_method(src, mask, use_mask, ds_method, fill_value, mode_rank, x_offset, y_offset, temp)
            return upsampling_method(temp, mask, use_mask, fill_value, x_offset, y_offset, out)
        else:
            return downsampling_method(src, mask, use_mask, ds_method, fill_value, mode_rank, x_offset, y_offset, out)
    elif out_w > src_wo or out_h > src_ho:
        return upsampling_method(src, mask, use_mask, fill_value, x_offset, y_offset, out)
    return src