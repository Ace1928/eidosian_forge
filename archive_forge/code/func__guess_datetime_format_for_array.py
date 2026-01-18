from __future__ import annotations
from collections import abc
from datetime import date
from functools import partial
from itertools import islice
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.tslibs import (
from pandas._libs.tslibs.conversion import cast_from_unit_vectorized
from pandas._libs.tslibs.parsing import (
from pandas._libs.tslibs.strptime import array_strptime
from pandas._typing import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.arrays import (
from pandas.core.algorithms import unique
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.datetimes import (
from pandas.core.construction import extract_array
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimes import DatetimeIndex
def _guess_datetime_format_for_array(arr, dayfirst: bool | None=False) -> str | None:
    if (first_non_null := tslib.first_non_null(arr)) != -1:
        if type((first_non_nan_element := arr[first_non_null])) is str:
            guessed_format = guess_datetime_format(first_non_nan_element, dayfirst=dayfirst)
            if guessed_format is not None:
                return guessed_format
            if tslib.first_non_null(arr[first_non_null + 1:]) != -1:
                warnings.warn('Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.', UserWarning, stacklevel=find_stack_level())
    return None