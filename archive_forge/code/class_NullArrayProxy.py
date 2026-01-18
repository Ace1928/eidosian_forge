from __future__ import annotations
import itertools
from typing import (
import numpy as np
from pandas._libs import (
from pandas.core.dtypes.astype import (
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
import pandas.core.algorithms as algos
from pandas.core.array_algos.quantile import quantile_compat
from pandas.core.array_algos.take import take_1d
from pandas.core.arrays import (
from pandas.core.construction import (
from pandas.core.indexers import (
from pandas.core.indexes.api import (
from pandas.core.indexes.base import get_values_for_csv
from pandas.core.internals.base import (
from pandas.core.internals.blocks import (
from pandas.core.internals.managers import make_na_array
class NullArrayProxy:
    """
    Proxy object for an all-NA array.

    Only stores the length of the array, and not the dtype. The dtype
    will only be known when actually concatenating (after determining the
    common dtype, for which this proxy is ignored).
    Using this object avoids that the internals/concat.py needs to determine
    the proper dtype and array type.
    """
    ndim = 1

    def __init__(self, n: int) -> None:
        self.n = n

    @property
    def shape(self) -> tuple[int]:
        return (self.n,)

    def to_array(self, dtype: DtypeObj) -> ArrayLike:
        """
        Helper function to create the actual all-NA array from the NullArrayProxy
        object.

        Parameters
        ----------
        arr : NullArrayProxy
        dtype : the dtype for the resulting array

        Returns
        -------
        np.ndarray or ExtensionArray
        """
        if isinstance(dtype, ExtensionDtype):
            empty = dtype.construct_array_type()._from_sequence([], dtype=dtype)
            indexer = -np.ones(self.n, dtype=np.intp)
            return empty.take(indexer, allow_fill=True)
        else:
            dtype = ensure_dtype_can_hold_na(dtype)
            fill_value = na_value_for_dtype(dtype)
            arr = np.empty(self.n, dtype=dtype)
            arr.fill(fill_value)
            return ensure_wrapped_if_datetimelike(arr)