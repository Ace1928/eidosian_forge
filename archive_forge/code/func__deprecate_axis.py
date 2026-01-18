from __future__ import annotations
import warnings
from collections.abc import Iterable
from types import BuiltinFunctionType
from typing import TYPE_CHECKING, Union
import numpy as np
import pandas
import pandas.core.common as com
import pandas.core.groupby
from pandas._libs import lib
from pandas.api.types import is_scalar
from pandas.core.apply import reconstruct_func
from pandas.core.dtypes.common import (
from pandas.errors import SpecificationError
from modin.core.dataframe.algebra.default2pandas.groupby import GroupBy
from modin.core.storage_formats.base.query_compiler import BaseQueryCompiler
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger, disable_logging
from modin.pandas.utils import cast_function_modin2pandas
from modin.utils import (
from .series import Series
from .utils import is_label
from .window import RollingGroupby
def _deprecate_axis(self, axis: int, name: str) -> None:
    if axis == 1:
        warnings.warn(f'{type(self).__name__}.{name} with axis=1 is deprecated and ' + 'will be removed in a future version. Operate on the un-grouped ' + 'DataFrame instead', FutureWarning)
    else:
        warnings.warn(f"The 'axis' keyword in {type(self).__name__}.{name} is deprecated " + 'and will be removed in a future version. ' + "Call without passing 'axis' instead.", FutureWarning)