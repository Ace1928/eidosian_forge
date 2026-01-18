from __future__ import annotations
import datetime
import warnings
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from os import PathLike
from typing import (
import numpy as np
import pandas as pd
from xarray.coding.calendar_ops import convert_calendar, interp_calendar
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core import alignment, computation, dtypes, indexing, ops, utils
from xarray.core._aggregations import DataArrayAggregations
from xarray.core.accessor_dt import CombinedDatetimelikeAccessor
from xarray.core.accessor_str import StringAccessor
from xarray.core.alignment import (
from xarray.core.arithmetic import DataArrayArithmetic
from xarray.core.common import AbstractArray, DataWithCoords, get_chunksizes
from xarray.core.computation import unify_chunks
from xarray.core.coordinates import (
from xarray.core.dataset import Dataset
from xarray.core.formatting import format_item
from xarray.core.indexes import (
from xarray.core.indexing import is_fancy_indexer, map_index_queries
from xarray.core.merge import PANDAS_TYPES, MergeError
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.types import (
from xarray.core.utils import (
from xarray.core.variable import (
from xarray.plot.accessor import DataArrayPlotAccessor
from xarray.plot.utils import _get_units_from_attrs
from xarray.util.deprecation_helpers import _deprecate_positional_args, deprecate_dims
Convert this array into a dask.dataframe.DataFrame.

        Parameters
        ----------
        dim_order : Sequence of Hashable or None , optional
            Hierarchical dimension order for the resulting dataframe.
            Array content is transposed to this order and then written out as flat
            vectors in contiguous order, so the last dimension in this list
            will be contiguous in the resulting DataFrame. This has a major influence
            on which operations are efficient on the resulting dask dataframe.
        set_index : bool, default: False
            If set_index=True, the dask DataFrame is indexed by this dataset's
            coordinate. Since dask DataFrames do not support multi-indexes,
            set_index only works if the dataset only contains one dimension.

        Returns
        -------
        dask.dataframe.DataFrame

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.arange(4 * 2 * 2).reshape(4, 2, 2),
        ...     dims=("time", "lat", "lon"),
        ...     coords={
        ...         "time": np.arange(4),
        ...         "lat": [-30, -20],
        ...         "lon": [120, 130],
        ...     },
        ...     name="eg_dataarray",
        ...     attrs={"units": "Celsius", "description": "Random temperature data"},
        ... )
        >>> da.to_dask_dataframe(["lat", "lon", "time"]).compute()
            lat  lon  time  eg_dataarray
        0   -30  120     0             0
        1   -30  120     1             4
        2   -30  120     2             8
        3   -30  120     3            12
        4   -30  130     0             1
        5   -30  130     1             5
        6   -30  130     2             9
        7   -30  130     3            13
        8   -20  120     0             2
        9   -20  120     1             6
        10  -20  120     2            10
        11  -20  120     3            14
        12  -20  130     0             3
        13  -20  130     1             7
        14  -20  130     2            11
        15  -20  130     3            15
        