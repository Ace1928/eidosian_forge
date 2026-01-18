from __future__ import annotations
from contextlib import suppress
import copy
from datetime import (
import itertools
import os
import re
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.lib import is_string_array
from pandas._libs.tslibs import timezones
from pandas.compat._optional import import_optional_dependency
from pandas.compat.pickle_compat import patch_pickle
from pandas.errors import (
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import array_equivalent
from pandas import (
from pandas.core.arrays import (
import pandas.core.common as com
from pandas.core.computation.pytables import (
from pandas.core.construction import extract_array
from pandas.core.indexes.api import ensure_index
from pandas.core.internals import (
from pandas.io.common import stringify_path
from pandas.io.formats.printing import (
def _set_tz(values: np.ndarray | Index, tz: str | tzinfo | None, coerce: bool=False) -> np.ndarray | DatetimeIndex:
    """
    coerce the values to a DatetimeIndex if tz is set
    preserve the input shape if possible

    Parameters
    ----------
    values : ndarray or Index
    tz : str or tzinfo
    coerce : if we do not have a passed timezone, coerce to M8[ns] ndarray
    """
    if isinstance(values, DatetimeIndex):
        assert values.tz is None or values.tz == tz
        if values.tz is not None:
            return values
    if tz is not None:
        if isinstance(values, DatetimeIndex):
            name = values.name
        else:
            name = None
            values = values.ravel()
        tz = _ensure_decoded(tz)
        values = DatetimeIndex(values, name=name)
        values = values.tz_localize('UTC').tz_convert(tz)
    elif coerce:
        values = np.asarray(values, dtype='M8[ns]')
    return values