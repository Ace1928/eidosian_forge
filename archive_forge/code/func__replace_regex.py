from __future__ import annotations
from functools import wraps
import inspect
import re
from typing import (
import warnings
import weakref
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.internals import (
from pandas._libs.missing import NA
from pandas._typing import (
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_bool_kwarg
from pandas.core.dtypes.astype import (
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import missing
import pandas.core.algorithms as algos
from pandas.core.array_algos.putmask import (
from pandas.core.array_algos.quantile import quantile_compat
from pandas.core.array_algos.replace import (
from pandas.core.array_algos.transforms import shift
from pandas.core.arrays import (
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.computation import expressions
from pandas.core.construction import (
from pandas.core.indexers import check_setitem_lengths
from pandas.core.indexes.base import get_values_for_csv
@final
def _replace_regex(self, to_replace, value, inplace: bool=False, mask=None, using_cow: bool=False, already_warned=None) -> list[Block]:
    """
        Replace elements by the given value.

        Parameters
        ----------
        to_replace : object or pattern
            Scalar to replace or regular expression to match.
        value : object
            Replacement object.
        inplace : bool, default False
            Perform inplace modification.
        mask : array-like of bool, optional
            True indicate corresponding element is ignored.
        using_cow: bool, default False
            Specifying if copy on write is enabled.

        Returns
        -------
        List[Block]
        """
    if not self._can_hold_element(to_replace):
        if using_cow:
            return [self.copy(deep=False)]
        return [self] if inplace else [self.copy()]
    rx = re.compile(to_replace)
    block = self._maybe_copy(using_cow, inplace)
    replace_regex(block.values, rx, value, mask)
    if inplace and warn_copy_on_write() and (already_warned is not None) and (not already_warned.warned_already):
        if self.refs.has_reference():
            warnings.warn(COW_WARNING_GENERAL_MSG, FutureWarning, stacklevel=find_stack_level())
            already_warned.warned_already = True
    nbs = block.convert(copy=False, using_cow=using_cow)
    opt = get_option('future.no_silent_downcasting')
    if (len(nbs) > 1 or nbs[0].dtype != block.dtype) and (not opt):
        warnings.warn("Downcasting behavior in `replace` is deprecated and will be removed in a future version. To retain the old behavior, explicitly call `result.infer_objects(copy=False)`. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`", FutureWarning, stacklevel=find_stack_level())
    return nbs