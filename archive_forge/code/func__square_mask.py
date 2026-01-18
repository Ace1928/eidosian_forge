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
def _square_mask(px):
    """Produce a square mask with sides of length ``2 * px + 1``"""
    px = int(px)
    w = 2 * px + 1
    return np.ones((w, w), dtype='bool')