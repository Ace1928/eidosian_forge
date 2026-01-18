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
def _nanargminmax(values: np.ndarray, mask: npt.NDArray[np.bool_], func) -> int:
    """
    See nanargminmax.__doc__.
    """
    idx = np.arange(values.shape[0])
    non_nans = values[~mask]
    non_nan_idx = idx[~mask]
    return non_nan_idx[func(non_nans)]