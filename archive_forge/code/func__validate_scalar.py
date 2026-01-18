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
def _validate_scalar(self, value, *, allow_listlike: bool=False, unbox: bool=True):
    """
        Validate that the input value can be cast to our scalar_type.

        Parameters
        ----------
        value : object
        allow_listlike: bool, default False
            When raising an exception, whether the message should say
            listlike inputs are allowed.
        unbox : bool, default True
            Whether to unbox the result before returning.  Note: unbox=False
            skips the setitem compatibility check.

        Returns
        -------
        self._scalar_type or NaT
        """
    if isinstance(value, self._scalar_type):
        pass
    elif isinstance(value, str):
        try:
            value = self._scalar_from_string(value)
        except ValueError as err:
            msg = self._validation_error_message(value, allow_listlike)
            raise TypeError(msg) from err
    elif is_valid_na_for_dtype(value, self.dtype):
        value = NaT
    elif isna(value):
        msg = self._validation_error_message(value, allow_listlike)
        raise TypeError(msg)
    elif isinstance(value, self._recognized_scalars):
        value = self._scalar_type(value)
    else:
        msg = self._validation_error_message(value, allow_listlike)
        raise TypeError(msg)
    if not unbox:
        return value
    return self._unbox_scalar(value)