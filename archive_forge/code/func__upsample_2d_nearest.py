from __future__ import annotations
from itertools import groupby
from math import floor, ceil
import dask.array as da
import numpy as np
from dask.delayed import delayed
from numba import prange
from .utils import ngjit, ngjit_parallel
@ngjit_parallel
def _upsample_2d_nearest(src, mask, use_mask, fill_value, x_offset, y_offset, out):
    src_w, src_h, out_w, out_h = _get_dimensions(src, out)
    x0_off, x1_off = x_offset
    y0_off, y1_off = y_offset
    src_w = src_w - x0_off - x1_off
    src_h = src_h - y0_off - y1_off
    if src_w == out_w and src_h == out_h:
        return src
    if out_w < src_w or out_h < src_h:
        raise ValueError('invalid target size')
    scale_x = src_w / out_w
    scale_y = src_h / out_h
    for out_y in prange(out_h):
        src_y = int(scale_y * out_y + y0_off)
        for out_x in range(out_w):
            src_x = int(scale_x * out_x + x0_off)
            value = src[src_y, src_x]
            if np.isfinite(value) and (not (use_mask and mask[src_y, src_x])):
                out[out_y, out_x] = value
            else:
                out[out_y, out_x] = fill_value
    return out