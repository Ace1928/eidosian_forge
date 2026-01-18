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
def _loc_array(self, iindexer, cindexer):
    iindexer_series = iindexer.to_dask_dataframe('_', self.obj.index)
    return self._loc_series(iindexer_series, cindexer)