from __future__ import annotations
from datetime import (
from typing import TYPE_CHECKING
import warnings
import numpy as np
from pandas._libs import index as libindex
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import OFFSET_TO_PERIOD_FREQSTR
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import is_integer
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import is_valid_na_for_dtype
from pandas.core.arrays.period import (
import pandas.core.common as com
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import maybe_extract_name
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin
from pandas.core.indexes.datetimes import (
from pandas.core.indexes.extension import inherit_names
def _cast_partial_indexing_scalar(self, label: datetime) -> Period:
    try:
        period = Period(label, freq=self.freq)
    except ValueError as err:
        raise KeyError(label) from err
    return period