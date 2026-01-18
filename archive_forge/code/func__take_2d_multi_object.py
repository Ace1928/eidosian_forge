from __future__ import annotations
import functools
from typing import (
import numpy as np
from pandas._libs import (
from pandas.core.dtypes.cast import maybe_promote
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import na_value_for_dtype
from pandas.core.construction import ensure_wrapped_if_datetimelike
def _take_2d_multi_object(arr: np.ndarray, indexer: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]], out: np.ndarray, fill_value, mask_info) -> None:
    row_idx, col_idx = indexer
    if mask_info is not None:
        (row_mask, col_mask), (row_needs, col_needs) = mask_info
    else:
        row_mask = row_idx == -1
        col_mask = col_idx == -1
        row_needs = row_mask.any()
        col_needs = col_mask.any()
    if fill_value is not None:
        if row_needs:
            out[row_mask, :] = fill_value
        if col_needs:
            out[:, col_mask] = fill_value
    for i, u_ in enumerate(row_idx):
        if u_ != -1:
            for j, v in enumerate(col_idx):
                if v != -1:
                    out[i, j] = arr[u_, v]