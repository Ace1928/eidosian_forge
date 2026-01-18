from __future__ import annotations
from itertools import count
import logging
from typing import TYPE_CHECKING
from toolz import unique, concat, pluck, get, memoize
from numba import literal_unroll
import numpy as np
import xarray as xr
from .antialias import AntialiasCombination
from .reductions import SpecialColumn, UsesCudaMutex, by, category_codes, summary
from .utils import (isnull, ngjit,
@ngjit
def aa_stage_2_copy_back(aggs_and_copies):
    for agg_and_copy in literal_unroll(aggs_and_copies):
        agg_and_copy[0][:] = agg_and_copy[1][:]