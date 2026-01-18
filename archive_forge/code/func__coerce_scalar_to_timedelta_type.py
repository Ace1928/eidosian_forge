from __future__ import annotations
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._libs.tslibs.timedeltas import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas.core.dtypes.generic import (
from pandas.core.arrays.timedeltas import sequence_to_td64ns
def _coerce_scalar_to_timedelta_type(r, unit: UnitChoices | None='ns', errors: DateTimeErrorChoices='raise'):
    """Convert string 'r' to a timedelta object."""
    result: Timedelta | NaTType
    try:
        result = Timedelta(r, unit)
    except ValueError:
        if errors == 'raise':
            raise
        if errors == 'ignore':
            return r
        result = NaT
    return result