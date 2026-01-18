from __future__ import annotations
import math
import re
import warnings
from datetime import timedelta
import numpy as np
import pandas as pd
from packaging.version import Version
from xarray.coding.times import (
from xarray.core.common import _contains_cftime_datetimes
from xarray.core.options import OPTIONS
from xarray.core.utils import is_scalar
def _parse_array_of_cftime_strings(strings, date_type):
    """Create a numpy array from an array of strings.

    For use in generating dates from strings for use with interp.  Assumes the
    array is either 0-dimensional or 1-dimensional.

    Parameters
    ----------
    strings : array of strings
        Strings to convert to dates
    date_type : cftime.datetime type
        Calendar type to use for dates

    Returns
    -------
    np.array
    """
    return np.array([_parse_iso8601_without_reso(date_type, s) for s in strings.ravel()]).reshape(strings.shape)