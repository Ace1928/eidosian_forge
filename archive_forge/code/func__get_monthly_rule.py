from __future__ import annotations
import numpy as np
import pandas as pd
from xarray.coding.cftime_offsets import _MONTH_ABBREVIATIONS, _legacy_to_new_freq
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core.common import _contains_datetime_like_objects
def _get_monthly_rule(self):
    if len(self.month_deltas) > 1:
        return None
    return {'cs': 'MS', 'ce': 'ME'}.get(month_anchor_check(self.index))