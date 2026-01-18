from __future__ import annotations
from collections import abc
from typing import (
import numpy as np
from numpy import ma
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
from pandas.core import (
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.string_ import StringDtype
from pandas.core.construction import (
from pandas.core.indexes.api import (
from pandas.core.internals.array_manager import (
from pandas.core.internals.blocks import (
from pandas.core.internals.managers import (
def _validate_or_indexify_columns(content: list[np.ndarray], columns: Index | None) -> Index:
    """
    If columns is None, make numbers as column names; Otherwise, validate that
    columns have valid length.

    Parameters
    ----------
    content : list of np.ndarrays
    columns : Index or None

    Returns
    -------
    Index
        If columns is None, assign positional column index value as columns.

    Raises
    ------
    1. AssertionError when content is not composed of list of lists, and if
        length of columns is not equal to length of content.
    2. ValueError when content is list of lists, but length of each sub-list
        is not equal
    3. ValueError when content is list of lists, but length of sub-list is
        not equal to length of content
    """
    if columns is None:
        columns = default_index(len(content))
    else:
        is_mi_list = isinstance(columns, list) and all((isinstance(col, list) for col in columns))
        if not is_mi_list and len(columns) != len(content):
            raise AssertionError(f'{len(columns)} columns passed, passed data had {len(content)} columns')
        if is_mi_list:
            if len({len(col) for col in columns}) > 1:
                raise ValueError('Length of columns passed for MultiIndex columns is different')
            if columns and len(columns[0]) != len(content):
                raise ValueError(f'{len(columns[0])} columns passed, passed data had {len(content)} columns')
    return columns