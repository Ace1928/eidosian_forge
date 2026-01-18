from __future__ import annotations
from datetime import (
from typing import TYPE_CHECKING
import warnings
import numpy as np
from pandas._libs.lib import is_list_like
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import notna
def _guess_time_format_for_array(arr):
    non_nan_elements = notna(arr).nonzero()[0]
    if len(non_nan_elements):
        element = arr[non_nan_elements[0]]
        for time_format in _time_formats:
            try:
                datetime.strptime(element, time_format)
                return time_format
            except ValueError:
                pass
    return None