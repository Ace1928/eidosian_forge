from __future__ import annotations
from datetime import (
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import abbrev_to_npy_unit
from pandas.errors import PerformanceWarning
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_inclusive
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
from pandas.core.arrays import datetimelike as dtl
from pandas.core.arrays._ranges import generate_regular_range
import pandas.core.common as com
from pandas.tseries.frequencies import get_period_alias
from pandas.tseries.offsets import (
def _field_accessor(name: str, field: str, docstring: str | None=None):

    def f(self):
        values = self._local_timestamps()
        if field in self._bool_ops:
            result: np.ndarray
            if field.endswith(('start', 'end')):
                freq = self.freq
                month_kw = 12
                if freq:
                    kwds = freq.kwds
                    month_kw = kwds.get('startingMonth', kwds.get('month', 12))
                result = fields.get_start_end_field(values, field, self.freqstr, month_kw, reso=self._creso)
            else:
                result = fields.get_date_field(values, field, reso=self._creso)
            return result
        if field in self._object_ops:
            result = fields.get_date_name_field(values, field, reso=self._creso)
            result = self._maybe_mask_results(result, fill_value=None)
        else:
            result = fields.get_date_field(values, field, reso=self._creso)
            result = self._maybe_mask_results(result, fill_value=None, convert='float64')
        return result
    f.__name__ = name
    f.__doc__ = docstring
    return property(f)