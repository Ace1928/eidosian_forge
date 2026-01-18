from __future__ import annotations
from collections.abc import Iterable
from typing import (
import numpy as np
from pandas.util._decorators import (
from pandas.core.dtypes.common import (
def _make_mask_from_list(self, args: Iterable[int]) -> bool | np.ndarray:
    positive = [arg for arg in args if arg >= 0]
    negative = [-arg - 1 for arg in args if arg < 0]
    mask: bool | np.ndarray = False
    if positive:
        mask |= np.isin(self._ascending_count, positive)
    if negative:
        mask |= np.isin(self._descending_count, negative)
    return mask