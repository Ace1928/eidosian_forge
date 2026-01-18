import ast
import hashlib
import re
import warnings
from collections.abc import Iterable
from typing import Hashable, List
import numpy as np
import pandas
from pandas._libs import lib
from pandas.api.types import is_scalar
from pandas.core.apply import reconstruct_func
from pandas.core.common import is_bool_indexer
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import (
from pandas.core.groupby.base import transformation_kernels
from pandas.core.indexes.api import ensure_index_from_sequences
from pandas.core.indexing import check_bool_indexer
from pandas.errors import DataError
from modin.config import CpuCount, RangePartitioning, use_range_partitioning_groupby
from modin.core.dataframe.algebra import (
from modin.core.dataframe.algebra.default2pandas.groupby import (
from modin.core.dataframe.pandas.metadata import (
from modin.core.storage_formats import BaseQueryCompiler
from modin.error_message import ErrorMessage
from modin.logging import get_logger
from modin.utils import (
from .aggregations import CorrCovBuilder
from .groupby import GroupbyReduceImpl, PivotTableImpl
from .merge import MergeImpl
from .utils import get_group_names, merge_partitioning
def _nancorr(self, min_periods=1, cov=False, ddof=1):
    """
        Compute either pairwise covariance or pairwise correlation of columns.

        This function considers NA/null values the same like pandas does.

        Parameters
        ----------
        min_periods : int, default: 1
            Minimum number of observations required per pair of columns
            to have a valid result.
        cov : boolean, default: False
            Either covariance or correlation should be computed.
        ddof : int, default: 1
            Means Delta Degrees of Freedom. The divisor used in calculations.

        Returns
        -------
        PandasQueryCompiler
            The covariance or correlation matrix.

        Notes
        -----
        This method is only used to compute covariance at the moment.
        """
    other = self.to_numpy()
    try:
        other_mask = self._isfinite().to_numpy()
    except TypeError as err:
        raise ValueError("Unsupported types with 'numeric_only=False'") from err
    n_cols = other.shape[1]
    if min_periods is None:
        min_periods = 1

    def map_func(df):
        """Compute covariance or correlation matrix for the passed frame."""
        df = df.to_numpy()
        n_rows = df.shape[0]
        df_mask = np.isfinite(df)
        result = np.empty((n_rows, n_cols), dtype=np.float64)
        for i in range(n_rows):
            df_ith_row = df[i]
            df_ith_mask = df_mask[i]
            for j in range(n_cols):
                other_jth_col = other[:, j]
                valid = df_ith_mask & other_mask[:, j]
                vx = df_ith_row[valid]
                vy = other_jth_col[valid]
                nobs = len(vx)
                if nobs < min_periods:
                    result[i, j] = np.nan
                else:
                    vx = vx - vx.mean()
                    vy = vy - vy.mean()
                    sumxy = (vx * vy).sum()
                    sumxx = (vx * vx).sum()
                    sumyy = (vy * vy).sum()
                    denom = nobs - ddof if cov else np.sqrt(sumxx * sumyy)
                    if denom != 0:
                        result[i, j] = sumxy / denom
                    else:
                        result[i, j] = np.nan
        return pandas.DataFrame(result)
    columns = self.columns
    index = columns.copy()
    transponed_self = self.transpose()
    new_modin_frame = transponed_self._modin_frame.apply_full_axis(1, map_func, new_index=index, new_columns=columns)
    return transponed_self.__constructor__(new_modin_frame)