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
def _initialize_chunksize(self, chunksize: int | None) -> int:
    if chunksize is None:
        return _DEFAULT_CHUNKSIZE_CELLS // (len(self.cols) or 1) or 1
    return int(chunksize)