from __future__ import annotations
from collections.abc import (
import datetime
from functools import partial
from typing import (
import uuid
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.lib import is_range_indexer
from pandas._typing import (
from pandas.errors import MergeError
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas import (
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.arrays.string_ import StringDtype
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.frame import _merge_doc
from pandas.core.indexes.api import default_index
from pandas.core.sorting import (
def _validate_tolerance(self, left_join_keys: list[ArrayLike]) -> None:
    if self.tolerance is not None:
        if self.left_index:
            lt = self.left.index._values
        else:
            lt = left_join_keys[-1]
        msg = f'incompatible tolerance {self.tolerance}, must be compat with type {repr(lt.dtype)}'
        if needs_i8_conversion(lt.dtype) or (isinstance(lt, ArrowExtensionArray) and lt.dtype.kind in 'mM'):
            if not isinstance(self.tolerance, datetime.timedelta):
                raise MergeError(msg)
            if self.tolerance < Timedelta(0):
                raise MergeError('tolerance must be positive')
        elif is_integer_dtype(lt.dtype):
            if not is_integer(self.tolerance):
                raise MergeError(msg)
            if self.tolerance < 0:
                raise MergeError('tolerance must be positive')
        elif is_float_dtype(lt.dtype):
            if not is_number(self.tolerance):
                raise MergeError(msg)
            if self.tolerance < 0:
                raise MergeError('tolerance must be positive')
        else:
            raise MergeError('key must be integer, timestamp or float')