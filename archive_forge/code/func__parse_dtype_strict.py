from __future__ import annotations
from datetime import (
from decimal import Decimal
import re
from typing import (
import warnings
import numpy as np
import pytz
from pandas._libs import (
from pandas._libs.interval import Interval
from pandas._libs.properties import cache_readonly
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import (
from pandas._libs.tslibs.offsets import BDay
from pandas.compat import pa_version_under10p1
from pandas.errors import PerformanceWarning
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import (
from pandas.util import capitalize_first_letter
@classmethod
def _parse_dtype_strict(cls, freq: str_type) -> BaseOffset:
    if isinstance(freq, str):
        if freq.startswith(('Period[', 'period[')):
            m = cls._match.search(freq)
            if m is not None:
                freq = m.group('freq')
        freq_offset = to_offset(freq, is_period=True)
        if freq_offset is not None:
            return freq_offset
    raise TypeError(f'PeriodDtype argument should be string or BaseOffset, got {type(freq).__name__}')