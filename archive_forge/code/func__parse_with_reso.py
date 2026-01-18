from __future__ import annotations
from abc import (
from typing import (
import warnings
import numpy as np
from pandas._config import using_copy_on_write
from pandas._libs import (
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas.core.arrays import (
from pandas.core.arrays.datetimelike import DatetimeLikeArrayMixin
import pandas.core.common as com
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import (
from pandas.core.indexes.extension import NDArrayBackedExtensionIndex
from pandas.core.indexes.range import RangeIndex
from pandas.core.tools.timedeltas import to_timedelta
def _parse_with_reso(self, label: str):
    try:
        if self.freq is None or hasattr(self.freq, 'rule_code'):
            freq = self.freq
    except NotImplementedError:
        freq = getattr(self, 'freqstr', getattr(self, 'inferred_freq', None))
    freqstr: str | None
    if freq is not None and (not isinstance(freq, str)):
        freqstr = freq.rule_code
    else:
        freqstr = freq
    if isinstance(label, np.str_):
        label = str(label)
    parsed, reso_str = parsing.parse_datetime_string_with_reso(label, freqstr)
    reso = Resolution.from_attrname(reso_str)
    return (parsed, reso)