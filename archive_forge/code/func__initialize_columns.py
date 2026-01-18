from __future__ import annotations
from collections.abc import (
import csv as csvlib
import os
from typing import (
import numpy as np
from pandas._libs import writers as libwriters
from pandas._typing import SequenceNotStr
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import notna
from pandas.core.indexes.api import Index
from pandas.io.common import get_handle
def _initialize_columns(self, cols: Iterable[Hashable] | None) -> npt.NDArray[np.object_]:
    if self.has_mi_columns:
        if cols is not None:
            msg = 'cannot specify cols with a MultiIndex on the columns'
            raise TypeError(msg)
    if cols is not None:
        if isinstance(cols, ABCIndex):
            cols = cols._get_values_for_csv(**self._number_format)
        else:
            cols = list(cols)
        self.obj = self.obj.loc[:, cols]
    new_cols = self.obj.columns
    return new_cols._get_values_for_csv(**self._number_format)