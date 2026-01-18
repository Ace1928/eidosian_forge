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
def _cast_inplace(terms, acceptable_dtypes, dtype) -> None:
    """
    Cast an expression inplace.

    Parameters
    ----------
    terms : Op
        The expression that should cast.
    acceptable_dtypes : list of acceptable numpy.dtype
        Will not cast if term's dtype in this list.
    dtype : str or numpy.dtype
        The dtype to cast to.
    """
    dt = np.dtype(dtype)
    for term in terms:
        if term.type in acceptable_dtypes:
            continue
        try:
            new_value = term.value.astype(dt)
        except AttributeError:
            new_value = dt.type(term.value)
        term.update(new_value)