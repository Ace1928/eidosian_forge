from typing import Any, Optional
from collections.abc import Mapping
import numpy as np
import pandas as pd
def _right_squeeze(arr, stop_dim=0):
    """
    Remove trailing singleton dimensions

    Parameters
    ----------
    arr : ndarray
        Input array
    stop_dim : int
        Dimension where checking should stop so that shape[i] is not checked
        for i < stop_dim

    Returns
    -------
    squeezed : ndarray
        Array with all trailing singleton dimensions (0 or 1) removed.
        Singleton dimensions for dimension < stop_dim are retained.
    """
    last = arr.ndim
    for s in reversed(arr.shape):
        if s > 1:
            break
        last -= 1
    last = max(last, stop_dim)
    return arr.reshape(arr.shape[:last])