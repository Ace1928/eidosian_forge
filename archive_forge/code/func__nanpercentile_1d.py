from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from pandas.core.dtypes.missing import (
def _nanpercentile_1d(values: np.ndarray, mask: npt.NDArray[np.bool_], qs: npt.NDArray[np.float64], na_value: Scalar, interpolation: str) -> Scalar | np.ndarray:
    """
    Wrapper for np.percentile that skips missing values, specialized to
    1-dimensional case.

    Parameters
    ----------
    values : array over which to find quantiles
    mask : ndarray[bool]
        locations in values that should be considered missing
    qs : np.ndarray[float64] of quantile indices to find
    na_value : scalar
        value to return for empty or all-null values
    interpolation : str

    Returns
    -------
    quantiles : scalar or array
    """
    values = values[~mask]
    if len(values) == 0:
        return np.full(len(qs), na_value)
    return np.percentile(values, qs, method=interpolation)