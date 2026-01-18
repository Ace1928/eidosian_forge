from enum import Enum
from typing import TYPE_CHECKING, Callable, Tuple
import numpy as np
import pandas
from pandas.core.dtypes.common import is_numeric_dtype
from modin.utils import MODIN_UNNAMED_SERIES_LABEL

        Build correlation matrix for a DataFrame that didn't have NaN values in it.

        Parameters
        ----------
        sum_of_pairwise_mul : pandas.DataFrame
        means : pandas.Series
        sums : pandas.Series
        count : int
        std : pandas.Series
        cols : pandas.Index

        Returns
        -------
        pandas.DataFrame
            Correlation matrix.
        