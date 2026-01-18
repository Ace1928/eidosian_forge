from __future__ import annotations
import numbers
import string
from typing import TYPE_CHECKING
import numpy as np
from pandas.core.dtypes.base import ExtensionDtype
import pandas as pd
from pandas.api.types import (
from pandas.core.arrays import ExtensionArray
class ListDtype(ExtensionDtype):
    type = list
    name = 'list'
    na_value = np.nan

    @classmethod
    def construct_array_type(cls) -> type_t[ListArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return ListArray