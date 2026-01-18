from __future__ import annotations
from contextlib import contextmanager
import copy
from functools import partial
import operator
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
import pandas as pd
from pandas import (
import pandas.core.common as com
from pandas.core.frame import (
from pandas.core.generic import NDFrame
from pandas.core.shared_docs import _shared_docs
from pandas.io.formats.format import save_to_buffer
from pandas.io.formats.style_render import (
def _highlight_between(data: NDFrame, props: str, left: Scalar | Sequence | np.ndarray | NDFrame | None=None, right: Scalar | Sequence | np.ndarray | NDFrame | None=None, inclusive: bool | str=True) -> np.ndarray:
    """
    Return an array of css props based on condition of data values within given range.
    """
    if np.iterable(left) and (not isinstance(left, str)):
        left = _validate_apply_axis_arg(left, 'left', None, data)
    if np.iterable(right) and (not isinstance(right, str)):
        right = _validate_apply_axis_arg(right, 'right', None, data)
    if inclusive == 'both':
        ops = (operator.ge, operator.le)
    elif inclusive == 'neither':
        ops = (operator.gt, operator.lt)
    elif inclusive == 'left':
        ops = (operator.ge, operator.lt)
    elif inclusive == 'right':
        ops = (operator.gt, operator.le)
    else:
        raise ValueError(f"'inclusive' values can be 'both', 'left', 'right', or 'neither' got {inclusive}")
    g_left = ops[0](data, left) if left is not None else np.full(data.shape, True, dtype=bool)
    if isinstance(g_left, (DataFrame, Series)):
        g_left = g_left.where(pd.notna(g_left), False)
    l_right = ops[1](data, right) if right is not None else np.full(data.shape, True, dtype=bool)
    if isinstance(l_right, (DataFrame, Series)):
        l_right = l_right.where(pd.notna(l_right), False)
    return np.where(g_left & l_right, props, '')