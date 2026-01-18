from typing import TYPE_CHECKING, Callable, Optional, Union
import numpy as np
import pandas
from pandas._typing import IndexLabel
from pandas.core.dtypes.cast import find_common_type
from modin.error_message import ErrorMessage
def _normalize_self_levels(self, reference=None):
    """
        Call ``self._normalize_levels()`` for known and unknown dtypes of this object.

        Parameters
        ----------
        reference : pandas.Index, optional
        """
    _, old_to_new_mapping = self._normalize_levels(self._known_dtypes.keys(), reference)
    for old_col, new_col in old_to_new_mapping.items():
        value = self._known_dtypes.pop(old_col)
        self._known_dtypes[new_col] = value
    self._cols_with_unknown_dtypes, _ = self._normalize_levels(self._cols_with_unknown_dtypes, reference)