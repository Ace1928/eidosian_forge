from __future__ import annotations
from csv import QUOTE_NONNUMERIC
from functools import partial
import operator
from shutil import get_terminal_size
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import (
from pandas._libs.arrays import NDArrayBacked
from pandas.compat.numpy import function as nv
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_bool_kwarg
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.accessor import (
from pandas.core.algorithms import (
from pandas.core.arrays._mixins import (
from pandas.core.base import (
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.ops.common import unpack_zerodim_and_defer
from pandas.core.sorting import nargsort
from pandas.core.strings.object_array import ObjectStringArrayMixin
from pandas.io.formats import console
def _cat_compare_op(op):
    opname = f'__{op.__name__}__'
    fill_value = op is operator.ne

    @unpack_zerodim_and_defer(opname)
    def func(self, other):
        hashable = is_hashable(other)
        if is_list_like(other) and len(other) != len(self) and (not hashable):
            raise ValueError('Lengths must match.')
        if not self.ordered:
            if opname in ['__lt__', '__gt__', '__le__', '__ge__']:
                raise TypeError('Unordered Categoricals can only compare equality or not')
        if isinstance(other, Categorical):
            msg = "Categoricals can only be compared if 'categories' are the same."
            if not self._categories_match_up_to_permutation(other):
                raise TypeError(msg)
            if not self.ordered and (not self.categories.equals(other.categories)):
                other_codes = recode_for_categories(other.codes, other.categories, self.categories, copy=False)
            else:
                other_codes = other._codes
            ret = op(self._codes, other_codes)
            mask = (self._codes == -1) | (other_codes == -1)
            if mask.any():
                ret[mask] = fill_value
            return ret
        if hashable:
            if other in self.categories:
                i = self._unbox_scalar(other)
                ret = op(self._codes, i)
                if opname not in {'__eq__', '__ge__', '__gt__'}:
                    mask = self._codes == -1
                    ret[mask] = fill_value
                return ret
            else:
                return ops.invalid_comparison(self, other, op)
        else:
            if opname not in ['__eq__', '__ne__']:
                raise TypeError(f"Cannot compare a Categorical for op {opname} with type {type(other)}.\nIf you want to compare values, use 'np.asarray(cat) <op> other'.")
            if isinstance(other, ExtensionArray) and needs_i8_conversion(other.dtype):
                return op(other, self)
            return getattr(np.array(self), opname)(np.array(other))
    func.__name__ = opname
    return func