from __future__ import annotations
import datetime
import typing
import numpy as np
import pandas as pd
from xarray.coding.cftime_offsets import (
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core.types import SideOptions
def _ceil_via_cftimeindex(date: CFTimeDatetime, freq: str | BaseCFTimeOffset):
    index = CFTimeIndex([date])
    return index.ceil(freq).item()