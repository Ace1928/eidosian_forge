from __future__ import annotations
import warnings
from collections.abc import Iterable
import numpy as np
import pandas as pd
from pandas.api.types import is_scalar, union_categoricals
from dask.array.core import Array
from dask.array.dispatch import percentile_lookup
from dask.array.percentile import _percentile
from dask.backends import CreationDispatch, DaskBackendEntrypoint
from dask.dataframe._compat import PANDAS_GE_220, is_any_real_numeric_dtype
from dask.dataframe.core import DataFrame, Index, Scalar, Series, _Frame
from dask.dataframe.dispatch import (
from dask.dataframe.extensions import make_array_nonempty, make_scalar
from dask.dataframe.utils import (
from dask.sizeof import SimpleSizeof, sizeof
from dask.utils import is_arraylike, is_series_like, typename
class ShuffleGroupResult(SimpleSizeof, dict):

    def __sizeof__(self) -> int:
        """
        The result of the shuffle split are typically small dictionaries
        (#keys << 100; typically <= 32) The splits are often non-uniformly
        distributed. Some of the splits may even be empty. Sampling the
        dictionary for size estimation can cause severe errors.

        See also https://github.com/dask/distributed/issues/4962
        """
        total_size = super().__sizeof__()
        for k, df in self.items():
            total_size += sizeof(k)
            total_size += sizeof(df)
        return total_size