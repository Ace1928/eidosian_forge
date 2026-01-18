from __future__ import annotations
import bisect
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype
from dask.array.core import Array
from dask.base import tokenize
from dask.dataframe import methods
from dask.dataframe._compat import IndexingError
from dask.dataframe.core import Series, new_dd_object
from dask.dataframe.utils import is_index_like, is_series_like, meta_nonempty
from dask.highlevelgraph import HighLevelGraph
from dask.utils import is_arraylike
def _loc_series(self, iindexer, cindexer):
    if not is_bool_dtype(iindexer.dtype):
        raise KeyError('Cannot index with non-boolean dask Series. Try passing computed values instead (e.g. ``ddf.loc[iindexer.compute()]``)')
    meta = self._make_meta(iindexer, cindexer)
    return self.obj.map_partitions(methods.loc, iindexer, cindexer, token='loc-series', meta=meta)