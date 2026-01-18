from __future__ import annotations
from collections import defaultdict
from typing import (
import numpy as np
from pandas._libs import (
from pandas._libs.hashtable import unique_label_indices
from pandas.core.dtypes.common import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import isna
from pandas.core.construction import extract_array
def _int64_cut_off(shape) -> int:
    acc = 1
    for i, mul in enumerate(shape):
        acc *= int(mul)
        if not acc < lib.i8max:
            return i
    return len(shape)