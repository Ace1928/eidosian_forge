from __future__ import annotations
import ctypes
import re
from typing import Any
import numpy as np
from pandas.compat._optional import import_optional_dependency
from pandas.errors import SettingWithCopyError
import pandas as pd
from pandas.core.interchange.dataframe_protocol import (
from pandas.core.interchange.utils import (

    Set null values for the data according to the column null kind.

    Parameters
    ----------
    data : np.ndarray or pd.Series
        Data to set nulls in.
    col : Column
        Column object that describes the `data`.
    validity : tuple(Buffer, dtype) or None
        The return value of ``col.buffers()``. We do not access the ``col.buffers()``
        here to not take the ownership of the memory of buffer objects.
    allow_modify_inplace : bool, default: True
        Whether to modify the `data` inplace when zero-copy is possible (True) or always
        modify a copy of the `data` (False).

    Returns
    -------
    np.ndarray or pd.Series
        Data with the nulls being set.
    