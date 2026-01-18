from __future__ import annotations
from datetime import (
from functools import wraps
import operator
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.arrays import NDArrayBacked
from pandas._libs.tslibs import (
from pandas._libs.tslibs.fields import (
from pandas._libs.tslibs.np_datetime import compare_mismatched_resolutions
from pandas._libs.tslibs.timedeltas import get_unit_for_round
from pandas._libs.tslibs.timestamps import integer_op_not_supported
from pandas._typing import (
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.algorithms import (
from pandas.core.array_algos import datetimelike_accumulations
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._mixins import (
from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.integer import IntegerArray
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.indexers import (
from pandas.core.ops.common import unpack_zerodim_and_defer
from pandas.core.ops.invalid import (
from pandas.tseries import frequencies
def _validate_comparison_value(self, other):
    if isinstance(other, str):
        try:
            other = self._scalar_from_string(other)
        except (ValueError, IncompatibleFrequency):
            raise InvalidComparison(other)
    if isinstance(other, self._recognized_scalars) or other is NaT:
        other = self._scalar_type(other)
        try:
            self._check_compatible_with(other)
        except (TypeError, IncompatibleFrequency) as err:
            raise InvalidComparison(other) from err
    elif not is_list_like(other):
        raise InvalidComparison(other)
    elif len(other) != len(self):
        raise ValueError('Lengths must match')
    else:
        try:
            other = self._validate_listlike(other, allow_object=True)
            self._check_compatible_with(other)
        except (TypeError, IncompatibleFrequency) as err:
            if is_object_dtype(getattr(other, 'dtype', None)):
                pass
            else:
                raise InvalidComparison(other) from err
    return other