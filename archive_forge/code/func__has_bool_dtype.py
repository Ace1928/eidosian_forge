from __future__ import annotations
import operator
from typing import TYPE_CHECKING
import warnings
import numpy as np
from pandas._config import get_option
from pandas.util._exceptions import find_stack_level
from pandas.core import roperator
from pandas.core.computation.check import NUMEXPR_INSTALLED
def _has_bool_dtype(x):
    try:
        return x.dtype == bool
    except AttributeError:
        return isinstance(x, (bool, np.bool_))