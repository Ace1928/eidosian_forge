from __future__ import annotations
import os
import re
from inspect import getmro
import numba as nb
import numpy as np
import pandas as pd
from toolz import memoize
from xarray import DataArray
import dask.dataframe as dd
import datashader.datashape as datashape
@ngjit
def _nanfirst_n_impl(ret_pixel, other_pixel):
    """Single pixel implementation of nanfirst_n_in_place.
    ret_pixel and other_pixel are both 1D arrays of the same length.

    Walk along other_pixel a value at a time, find insertion index in
    ret_pixel and shift values along to insert.  Next other_pixel value is
    inserted at a higher index, so this walks the two pixel arrays just once
    each.
    """
    n = len(ret_pixel)
    istart = 0
    for other_value in other_pixel:
        if isnull(other_value):
            break
        else:
            for i in range(istart, n):
                if isnull(ret_pixel[i]):
                    ret_pixel[i] = other_value
                    istart = i + 1
                    break