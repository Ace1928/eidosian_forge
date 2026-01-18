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
def _add_arithmetic_ops(cls) -> None:
    setattr(cls, '__add__', cls._create_arithmetic_method(operator.add))
    setattr(cls, '__radd__', cls._create_arithmetic_method(roperator.radd))
    setattr(cls, '__sub__', cls._create_arithmetic_method(operator.sub))
    setattr(cls, '__rsub__', cls._create_arithmetic_method(roperator.rsub))
    setattr(cls, '__mul__', cls._create_arithmetic_method(operator.mul))
    setattr(cls, '__rmul__', cls._create_arithmetic_method(roperator.rmul))
    setattr(cls, '__pow__', cls._create_arithmetic_method(operator.pow))
    setattr(cls, '__rpow__', cls._create_arithmetic_method(roperator.rpow))
    setattr(cls, '__mod__', cls._create_arithmetic_method(operator.mod))
    setattr(cls, '__rmod__', cls._create_arithmetic_method(roperator.rmod))
    setattr(cls, '__floordiv__', cls._create_arithmetic_method(operator.floordiv))
    setattr(cls, '__rfloordiv__', cls._create_arithmetic_method(roperator.rfloordiv))
    setattr(cls, '__truediv__', cls._create_arithmetic_method(operator.truediv))
    setattr(cls, '__rtruediv__', cls._create_arithmetic_method(roperator.rtruediv))
    setattr(cls, '__divmod__', cls._create_arithmetic_method(divmod))
    setattr(cls, '__rdivmod__', cls._create_arithmetic_method(roperator.rdivmod))