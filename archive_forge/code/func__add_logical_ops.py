from __future__ import annotations
import operator
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas.compat import set_function_name
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import (
from pandas.core.dtypes.cast import maybe_cast_pointwise_result
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import isna
from pandas.core import (
from pandas.core.algorithms import (
from pandas.core.array_algos.quantile import quantile_with_mask
from pandas.core.missing import _fill_limit_area_1d
from pandas.core.sorting import (
@classmethod
def _add_logical_ops(cls) -> None:
    setattr(cls, '__and__', cls._create_logical_method(operator.and_))
    setattr(cls, '__rand__', cls._create_logical_method(roperator.rand_))
    setattr(cls, '__or__', cls._create_logical_method(operator.or_))
    setattr(cls, '__ror__', cls._create_logical_method(roperator.ror_))
    setattr(cls, '__xor__', cls._create_logical_method(operator.xor))
    setattr(cls, '__rxor__', cls._create_logical_method(roperator.rxor))