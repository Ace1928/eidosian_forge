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
def _initialize_index_label(self, index_label: IndexLabel | None) -> IndexLabel:
    if index_label is not False:
        if index_label is None:
            return self._get_index_label_from_obj()
        elif not isinstance(index_label, (list, tuple, np.ndarray, ABCIndex)):
            return [index_label]
    return index_label