from __future__ import annotations
from typing import (
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.generic import ABCDataFrame
from pandas.core.indexes.base import Index
class ExtensionIndex(Index):
    """
    Index subclass for indexes backed by ExtensionArray.
    """
    _data: IntervalArray | NDArrayBackedExtensionArray

    def _validate_fill_value(self, value):
        """
        Convert value to be insertable to underlying array.
        """
        return self._data._validate_setitem_value(value)

    @cache_readonly
    def _isnan(self) -> npt.NDArray[np.bool_]:
        return self._data.isna()