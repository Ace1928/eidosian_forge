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
@concat_dispatch.register_lazy('cudf')
@from_pyarrow_table_dispatch.register_lazy('cudf')
@group_split_dispatch.register_lazy('cudf')
@get_parallel_type.register_lazy('cudf')
@hash_object_dispatch.register_lazy('cudf')
@meta_nonempty.register_lazy('cudf')
@make_meta_dispatch.register_lazy('cudf')
@make_meta_obj.register_lazy('cudf')
@percentile_lookup.register_lazy('cudf')
@to_pyarrow_table_dispatch.register_lazy('cudf')
@tolist_dispatch.register_lazy('cudf')
def _register_cudf():
    import dask_cudf