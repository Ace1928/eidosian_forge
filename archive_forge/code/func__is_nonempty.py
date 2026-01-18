from __future__ import annotations
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.astype import astype_array
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas.core.dtypes.generic import (
def _is_nonempty(x, axis) -> bool:
    if x.ndim <= axis:
        return True
    return x.shape[axis] > 0