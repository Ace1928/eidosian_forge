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
class ExtensionArraySupportsAnyAll(ExtensionArray):

    def any(self, *, skipna: bool=True) -> bool:
        raise AbstractMethodError(self)

    def all(self, *, skipna: bool=True) -> bool:
        raise AbstractMethodError(self)