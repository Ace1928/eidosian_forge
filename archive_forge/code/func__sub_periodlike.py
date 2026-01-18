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
@final
def _sub_periodlike(self, other: Period | PeriodArray) -> npt.NDArray[np.object_]:
    if not isinstance(self.dtype, PeriodDtype):
        raise TypeError(f'cannot subtract {type(other).__name__} from {type(self).__name__}')
    self = cast('PeriodArray', self)
    self._check_compatible_with(other)
    other_i8, o_mask = self._get_i8_values_and_mask(other)
    new_i8_data = add_overflowsafe(self.asi8, np.asarray(-other_i8, dtype='i8'))
    new_data = np.array([self.freq.base * x for x in new_i8_data])
    if o_mask is None:
        mask = self._isnan
    else:
        mask = self._isnan | o_mask
    new_data[mask] = NaT
    return new_data