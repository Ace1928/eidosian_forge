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
def _finalize_columns_and_data(content: np.ndarray, columns: Index | None, dtype: DtypeObj | None) -> tuple[list[ArrayLike], Index]:
    """
    Ensure we have valid columns, cast object dtypes if possible.
    """
    contents = list(content.T)
    try:
        columns = _validate_or_indexify_columns(contents, columns)
    except AssertionError as err:
        raise ValueError(err) from err
    if len(contents) and contents[0].dtype == np.object_:
        contents = convert_object_array(contents, dtype=dtype)
    return (contents, columns)