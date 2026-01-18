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
def _fill_mask_inplace(self, method: str, limit: int | None, mask: npt.NDArray[np.bool_]) -> None:
    """
        Replace values in locations specified by 'mask' using pad or backfill.

        See also
        --------
        ExtensionArray.fillna
        """
    func = missing.get_fill_func(method)
    npvalues = self.astype(object)
    func(npvalues, limit=limit, mask=mask.copy())
    new_values = self._from_sequence(npvalues, dtype=self.dtype)
    self[mask] = new_values[mask]