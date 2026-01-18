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
def _get_antialias_stage_2_combine_func(combination: AntialiasCombination, zero: float, n_reduction: bool, categorical: bool):
    if n_reduction:
        if zero == -1:
            if combination in (AntialiasCombination.MAX, AntialiasCombination.LAST):
                return row_max_n_in_place_4d if categorical else row_max_n_in_place_3d
            elif combination in (AntialiasCombination.MIN, AntialiasCombination.FIRST):
                return row_min_n_in_place_4d if categorical else row_min_n_in_place_3d
            else:
                raise NotImplementedError
        elif combination == AntialiasCombination.MAX:
            return nanmax_n_in_place_4d if categorical else nanmax_n_in_place_3d
        elif combination == AntialiasCombination.MIN:
            return nanmin_n_in_place_4d if categorical else nanmin_n_in_place_3d
        elif combination == AntialiasCombination.FIRST:
            return nanfirst_n_in_place_4d if categorical else nanfirst_n_in_place_3d
        elif combination == AntialiasCombination.LAST:
            return nanlast_n_in_place_4d if categorical else nanlast_n_in_place_3d
        else:
            raise NotImplementedError
    elif zero == -1:
        if combination in (AntialiasCombination.MAX, AntialiasCombination.LAST):
            return row_max_in_place
        elif combination in (AntialiasCombination.MIN, AntialiasCombination.FIRST):
            return row_min_in_place
        else:
            raise NotImplementedError
    elif combination == AntialiasCombination.MAX:
        return nanmax_in_place
    elif combination == AntialiasCombination.MIN:
        return nanmin_in_place
    elif combination == AntialiasCombination.FIRST:
        return nanfirst_in_place
    elif combination == AntialiasCombination.LAST:
        return nanlast_in_place
    else:
        return nansum_in_place