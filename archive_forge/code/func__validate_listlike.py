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
def _validate_listlike(self, value, allow_object: bool=False):
    if isinstance(value, type(self)):
        if self.dtype.kind in 'mM' and (not allow_object):
            value = value.as_unit(self.unit, round_ok=False)
        return value
    if isinstance(value, list) and len(value) == 0:
        return type(self)._from_sequence([], dtype=self.dtype)
    if hasattr(value, 'dtype') and value.dtype == object:
        if lib.infer_dtype(value) in self._infer_matches:
            try:
                value = type(self)._from_sequence(value)
            except (ValueError, TypeError):
                if allow_object:
                    return value
                msg = self._validation_error_message(value, True)
                raise TypeError(msg)
    value = extract_array(value, extract_numpy=True)
    value = pd_array(value)
    value = extract_array(value, extract_numpy=True)
    if is_all_strings(value):
        try:
            value = type(self)._from_sequence(value, dtype=self.dtype)
        except ValueError:
            pass
    if isinstance(value.dtype, CategoricalDtype):
        if value.categories.dtype == self.dtype:
            value = value._internal_get_values()
            value = extract_array(value, extract_numpy=True)
    if allow_object and is_object_dtype(value.dtype):
        pass
    elif not type(self)._is_recognized_dtype(value.dtype):
        msg = self._validation_error_message(value, True)
        raise TypeError(msg)
    if self.dtype.kind in 'mM' and (not allow_object):
        value = value.as_unit(self.unit, round_ok=False)
    return value