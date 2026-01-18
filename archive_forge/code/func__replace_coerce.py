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
def _replace_coerce(self, to_replace, value, mask: npt.NDArray[np.bool_], inplace: bool=True, regex: bool=False, using_cow: bool=False) -> list[Block]:
    """
        Replace value corresponding to the given boolean array with another
        value.

        Parameters
        ----------
        to_replace : object or pattern
            Scalar to replace or regular expression to match.
        value : object
            Replacement object.
        mask : np.ndarray[bool]
            True indicate corresponding element is ignored.
        inplace : bool, default True
            Perform inplace modification.
        regex : bool, default False
            If true, perform regular expression substitution.

        Returns
        -------
        List[Block]
        """
    if should_use_regex(regex, to_replace):
        return self._replace_regex(to_replace, value, inplace=inplace, mask=mask)
    else:
        if value is None:
            if mask.any():
                has_ref = self.refs.has_reference()
                nb = self.astype(np.dtype(object), copy=False, using_cow=using_cow)
                if (nb is self or using_cow) and (not inplace):
                    nb = nb.copy()
                elif inplace and has_ref and nb.refs.has_reference() and using_cow:
                    nb = nb.copy()
                putmask_inplace(nb.values, mask, value)
                return [nb]
            if using_cow:
                return [self]
            return [self] if inplace else [self.copy()]
        return self.replace(to_replace=to_replace, value=value, inplace=inplace, mask=mask, using_cow=using_cow)