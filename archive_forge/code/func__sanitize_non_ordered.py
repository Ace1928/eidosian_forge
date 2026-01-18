from __future__ import annotations
from collections.abc import Sequence
from typing import (
import warnings
import numpy as np
from numpy import ma
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._typing import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import isna
import pandas.core.common as com
def _sanitize_non_ordered(data) -> None:
    """
    Raise only for unordered sets, e.g., not for dict_keys
    """
    if isinstance(data, (set, frozenset)):
        raise TypeError(f"'{type(data).__name__}' type is unordered")