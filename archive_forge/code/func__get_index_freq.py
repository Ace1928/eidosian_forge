from __future__ import annotations
import functools
from typing import (
import warnings
import numpy as np
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.converter import (
from pandas.tseries.frequencies import (
def _get_index_freq(index: Index) -> BaseOffset | None:
    freq = getattr(index, 'freq', None)
    if freq is None:
        freq = getattr(index, 'inferred_freq', None)
        if freq == 'B':
            weekdays = np.unique(index.dayofweek)
            if 5 in weekdays or 6 in weekdays:
                freq = None
    freq = to_offset(freq)
    return freq