import abc
import warnings
from typing import Hashable, List, Optional
import numpy as np
import pandas
import pandas.core.resample
from pandas._typing import DtypeBackend, IndexLabel, Suffixes
from pandas.core.dtypes.common import is_number, is_scalar
from modin.config import StorageFormat
from modin.core.dataframe.algebra.default2pandas import (
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, try_cast_to_pandas
from . import doc_utils
def _get_column_as_pandas_series(self, key):
    """
        Get column data by label as pandas.Series.

        Parameters
        ----------
        key : Any
            Column label.

        Returns
        -------
        pandas.Series
        """
    result = self.getitem_array([key]).to_pandas().squeeze(axis=1)
    if not isinstance(result, pandas.Series):
        raise RuntimeError(f'Expected getting column {key} to give ' + f'pandas.Series, but instead got {type(result)}')
    return result