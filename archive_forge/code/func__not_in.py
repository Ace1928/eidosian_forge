from __future__ import annotations
from datetime import datetime
from functools import partial
import operator
from typing import (
import numpy as np
from pandas._libs.tslibs import Timestamp
from pandas.core.dtypes.common import (
import pandas.core.common as com
from pandas.core.computation.common import (
from pandas.core.computation.scope import DEFAULT_GLOBALS
from pandas.io.formats.printing import (
def _not_in(x, y):
    """
    Compute the vectorized membership of ``x not in y`` if possible,
    otherwise use Python.
    """
    try:
        return ~x.isin(y)
    except AttributeError:
        if is_list_like(x):
            try:
                return ~y.isin(x)
            except AttributeError:
                pass
        return x not in y