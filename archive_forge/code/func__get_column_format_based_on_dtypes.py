from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas.core.dtypes.generic import ABCMultiIndex
def _get_column_format_based_on_dtypes(self) -> str:
    """Get column format based on data type.

        Right alignment for numbers and left - for strings.
        """

    def get_col_type(dtype) -> str:
        if issubclass(dtype.type, np.number):
            return 'r'
        return 'l'
    dtypes = self.frame.dtypes._values
    return ''.join(map(get_col_type, dtypes))