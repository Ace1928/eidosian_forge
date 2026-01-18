from __future__ import annotations
import logging # isort:skip
import datetime as dt
import uuid
from functools import lru_cache
from threading import Lock
from typing import TYPE_CHECKING, Any
import numpy as np
from ..core.types import ID
from ..settings import settings
from .strings import format_docstring
@lru_cache(None)
def _compute_datetime_types() -> set[type]:
    import pandas as pd
    result = {dt.time, dt.datetime, np.datetime64}
    result.add(pd.Timestamp)
    result.add(pd.Timedelta)
    result.add(pd.Period)
    result.add(type(pd.NaT))
    return result