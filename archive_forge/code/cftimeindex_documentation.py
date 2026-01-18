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
Round dates to a fixed frequency.

        Parameters
        ----------
        freq : str
            The frequency level to round the index to.  Must be a fixed
            frequency like 'S' (second) not 'ME' (month end).  See `frequency
            aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_
            for a list of possible values.

        Returns
        -------
        CFTimeIndex
        