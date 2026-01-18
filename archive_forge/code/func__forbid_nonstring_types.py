from __future__ import annotations
import codecs
from functools import wraps
import re
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._typing import (
from pandas.util._decorators import Appender
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import isna
from pandas.core.arrays import ExtensionArray
from pandas.core.base import NoNewAttributesMixin
from pandas.core.construction import extract_array
def _forbid_nonstring_types(func: F) -> F:
    func_name = func.__name__ if name is None else name

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self._inferred_dtype not in allowed_types:
            msg = f"Cannot use .str.{func_name} with values of inferred dtype '{self._inferred_dtype}'."
            raise TypeError(msg)
        return func(self, *args, **kwargs)
    wrapper.__name__ = func_name
    return cast(F, wrapper)