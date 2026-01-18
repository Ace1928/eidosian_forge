from __future__ import annotations
import numbers
import typing
import numpy as np
import pandas as pd
import pandas.api.types as pdtypes
from ..exceptions import PlotnineError
def _is_datetime_or_timedelta(value):
    return pd.Series(value).dtype.kind in ('M', 'm')