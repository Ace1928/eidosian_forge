from __future__ import annotations
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.missing import NA
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.internals.array_manager import ArrayManager
from pandas.core.internals.blocks import (
from pandas.core.internals.managers import (
class JoinUnit:

    def __init__(self, block: Block) -> None:
        self.block = block

    def __repr__(self) -> str:
        return f'{type(self).__name__}({repr(self.block)})'

    def _is_valid_na_for(self, dtype: DtypeObj) -> bool:
        """
        Check that we are all-NA of a type/dtype that is compatible with this dtype.
        Augments `self.is_na` with an additional check of the type of NA values.
        """
        if not self.is_na:
            return False
        blk = self.block
        if blk.dtype.kind == 'V':
            return True
        if blk.dtype == object:
            values = blk.values
            return all((is_valid_na_for_dtype(x, dtype) for x in values.ravel(order='K')))
        na_value = blk.fill_value
        if na_value is NaT and blk.dtype != dtype:
            return False
        if na_value is NA and needs_i8_conversion(dtype):
            return False
        return is_valid_na_for_dtype(na_value, dtype)

    @cache_readonly
    def is_na(self) -> bool:
        blk = self.block
        if blk.dtype.kind == 'V':
            return True
        if not blk._can_hold_na:
            return False
        values = blk.values
        if values.size == 0:
            return True
        if isinstance(values.dtype, SparseDtype):
            return False
        if values.ndim == 1:
            val = values[0]
            if not is_scalar(val) or not isna(val):
                return False
            return isna_all(values)
        else:
            val = values[0][0]
            if not is_scalar(val) or not isna(val):
                return False
            return all((isna_all(row) for row in values))

    @cache_readonly
    def is_na_after_size_and_isna_all_deprecation(self) -> bool:
        """
        Will self.is_na be True after values.size == 0 deprecation and isna_all
        deprecation are enforced?
        """
        blk = self.block
        if blk.dtype.kind == 'V':
            return True
        return False

    def get_reindexed_values(self, empty_dtype: DtypeObj, upcasted_na) -> ArrayLike:
        values: ArrayLike
        if upcasted_na is None and self.block.dtype.kind != 'V':
            return self.block.values
        else:
            fill_value = upcasted_na
            if self._is_valid_na_for(empty_dtype):
                blk_dtype = self.block.dtype
                if blk_dtype == np.dtype('object'):
                    values = cast(np.ndarray, self.block.values)
                    if values.size and values[0, 0] is None:
                        fill_value = None
                return make_na_array(empty_dtype, self.block.shape, fill_value)
            return self.block.values