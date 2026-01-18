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
def _loc(self, iindexer, cindexer):
    """Helper function for the .loc accessor"""
    if isinstance(iindexer, Series):
        return self._loc_series(iindexer, cindexer)
    elif isinstance(iindexer, Array):
        return self._loc_array(iindexer, cindexer)
    elif callable(iindexer):
        return self._loc(iindexer(self.obj), cindexer)
    if self.obj.known_divisions:
        iindexer = self._maybe_partial_time_string(iindexer)
        if isinstance(iindexer, slice):
            return self._loc_slice(iindexer, cindexer)
        elif is_series_like(iindexer) and (not is_bool_dtype(iindexer.dtype)):
            return self._loc_list(iindexer.values, cindexer)
        elif isinstance(iindexer, list) or is_arraylike(iindexer):
            return self._loc_list(iindexer, cindexer)
        else:
            return self._loc_element(iindexer, cindexer)
    else:
        if isinstance(iindexer, (list, np.ndarray)) or (is_series_like(iindexer) and (not is_bool_dtype(iindexer.dtype))):
            msg = 'Cannot index with list against unknown division. Try setting divisions using ``ddf.set_index``'
            raise KeyError(msg)
        elif not isinstance(iindexer, slice):
            iindexer = slice(iindexer, iindexer)
        meta = self._make_meta(iindexer, cindexer)
        return self.obj.map_partitions(methods.try_loc, iindexer, cindexer, meta=meta)