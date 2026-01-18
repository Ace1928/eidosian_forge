from __future__ import annotations
import numbers
from typing import TYPE_CHECKING
import numpy as np
from pandas.core.dtypes.base import ExtensionDtype
import pandas as pd
from pandas.core.arrays import ExtensionArray
class FloatAttrDtype(ExtensionDtype):
    type = float
    name = 'float_attr'
    na_value = np.nan

    @classmethod
    def construct_array_type(cls) -> type_t[FloatAttrArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return FloatAttrArray