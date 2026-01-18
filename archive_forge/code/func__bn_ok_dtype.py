from __future__ import annotations
import functools
import itertools
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import (
from pandas._typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import (
def _bn_ok_dtype(dtype: DtypeObj, name: str) -> bool:
    if dtype != object and (not needs_i8_conversion(dtype)):
        return name not in ['nansum', 'nanprod', 'nanmean']
    return False