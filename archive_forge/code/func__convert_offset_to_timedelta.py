from __future__ import annotations
import datetime
import typing
import numpy as np
import pandas as pd
from xarray.coding.cftime_offsets import (
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core.types import SideOptions
def _convert_offset_to_timedelta(offset: datetime.timedelta | str | BaseCFTimeOffset) -> datetime.timedelta:
    if isinstance(offset, datetime.timedelta):
        return offset
    elif isinstance(offset, (str, Tick)):
        return to_offset(offset).as_timedelta()
    else:
        raise ValueError