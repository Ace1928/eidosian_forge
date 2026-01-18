from __future__ import annotations
from collections.abc import Iterator
from io import BytesIO
import warnings
import numpy as np
import numba as nb
import toolz as tz
import xarray as xr
import dask.array as da
from PIL.Image import fromarray
from datashader.colors import rgb, Sets1to3
from datashader.utils import nansum_missing, ngjit
def _rescale_discrete_levels(discrete_levels, span):
    if discrete_levels is None:
        raise ValueError('interpolator did not return a valid discrete_levels')
    m = -0.5 / 98.0
    c = 1.5 - 2 * m
    multiple = m * discrete_levels + c
    if multiple > 1:
        lower_span = max(span[1] - multiple * (span[1] - span[0]), 0)
        span = (lower_span, 1)
    return span