from __future__ import annotations
import numpy as np
import pandas as pd
from xarray.coding.cftime_offsets import _MONTH_ABBREVIATIONS, _legacy_to_new_freq
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core.common import _contains_datetime_like_objects
def _maybe_add_count(base: str, count: float):
    """If count is greater than 1, add it to the base offset string"""
    if count != 1:
        assert count == int(count)
        count = int(count)
        return f'{count}{base}'
    else:
        return base