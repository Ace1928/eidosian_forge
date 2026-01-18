from enum import Enum
from typing import TYPE_CHECKING, Callable, Tuple
import numpy as np
import pandas
from pandas.core.dtypes.common import is_numeric_dtype
from modin.utils import MODIN_UNNAMED_SERIES_LABEL
@classmethod
def _build_map_reduce_methods(cls, min_periods: int, method: Method, numeric_only: bool) -> Tuple[Callable[[pandas.DataFrame], pandas.DataFrame], Callable[[pandas.DataFrame], pandas.DataFrame]]:
    """
        Build MapReduce kernels for the specified corr/cov method.

        Parameters
        ----------
        min_periods : int
            The parameter to pass to the reduce method.
        method : CorrCovBuilder.Method
            Whether the kernels compute correlation or covariance.
        numeric_only : bool
            Whether to only include numeric types.

        Returns
        -------
        Tuple[Callable(pandas.DataFrame) -> pandas.DataFrame, Callable(pandas.DataFrame) -> pandas.DataFrame]
            A tuple holding the Map (at the first position) and the Reduce (at the second position) kernels
            computing correlation/covariance matrix.
        """
    if method == cls.Method.COV:
        raise NotImplementedError('Computing covariance is not yet implemented.')
    return (lambda df: _CorrCovKernels.map(df, numeric_only), lambda df: _CorrCovKernels.reduce(df, min_periods, method))