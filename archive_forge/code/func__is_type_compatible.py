from __future__ import annotations
from operator import (
import textwrap
from typing import (
import numpy as np
from pandas._libs import lib
from pandas._libs.interval import (
from pandas._libs.tslibs import (
from pandas.errors import InvalidIndexError
from pandas.util._decorators import (
from pandas.util._exceptions import rewrite_exception
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import is_valid_na_for_dtype
from pandas.core.algorithms import unique
from pandas.core.arrays.datetimelike import validate_periods
from pandas.core.arrays.interval import (
import pandas.core.common as com
from pandas.core.indexers import is_valid_positional_slice
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import (
from pandas.core.indexes.datetimes import (
from pandas.core.indexes.extension import (
from pandas.core.indexes.multi import MultiIndex
from pandas.core.indexes.timedeltas import (
def _is_type_compatible(a, b) -> bool:
    """
    Helper for interval_range to check type compat of start/end/freq.
    """
    is_ts_compat = lambda x: isinstance(x, (Timestamp, BaseOffset))
    is_td_compat = lambda x: isinstance(x, (Timedelta, BaseOffset))
    return is_number(a) and is_number(b) or (is_ts_compat(a) and is_ts_compat(b)) or (is_td_compat(a) and is_td_compat(b)) or com.any_none(a, b)