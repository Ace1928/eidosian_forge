from __future__ import annotations
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.missing import NA
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.internals.array_manager import ArrayManager
from pandas.core.internals.blocks import (
from pandas.core.internals.managers import (
def _concatenate_array_managers(mgrs: list[ArrayManager], axes: list[Index], concat_axis: AxisInt) -> Manager2D:
    """
    Concatenate array managers into one.

    Parameters
    ----------
    mgrs_indexers : list of (ArrayManager, {axis: indexer,...}) tuples
    axes : list of Index
    concat_axis : int

    Returns
    -------
    ArrayManager
    """
    if concat_axis == 1:
        return mgrs[0].concat_vertical(mgrs, axes)
    else:
        assert concat_axis == 0
        return mgrs[0].concat_horizontal(mgrs, axes)