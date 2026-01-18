from __future__ import annotations
from functools import wraps
import inspect
import re
from typing import (
import warnings
import weakref
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.internals import (
from pandas._libs.missing import NA
from pandas._typing import (
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_bool_kwarg
from pandas.core.dtypes.astype import (
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import missing
import pandas.core.algorithms as algos
from pandas.core.array_algos.putmask import (
from pandas.core.array_algos.quantile import quantile_compat
from pandas.core.array_algos.replace import (
from pandas.core.array_algos.transforms import shift
from pandas.core.arrays import (
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.computation import expressions
from pandas.core.construction import (
from pandas.core.indexers import check_setitem_lengths
from pandas.core.indexes.base import get_values_for_csv
@final
@maybe_split
def _downcast_2d(self, dtype, using_cow: bool=False) -> list[Block]:
    """
        downcast specialized to 2D case post-validation.

        Refactored to allow use of maybe_split.
        """
    new_values = maybe_downcast_to_dtype(self.values, dtype=dtype)
    new_values = maybe_coerce_values(new_values)
    refs = self.refs if new_values is self.values else None
    return [self.make_block(new_values, refs=refs)]