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
def _loc_list(self, iindexer, cindexer):
    name = 'loc-%s' % tokenize(iindexer, self.obj)
    parts = self._get_partitions(iindexer)
    meta = self._make_meta(iindexer, cindexer)
    if len(iindexer):
        dsk = {}
        divisions = []
        items = sorted(parts.items())
        for i, (div, indexer) in enumerate(items):
            dsk[name, i] = (methods.loc, (self._name, div), indexer, cindexer)
            divisions.append(sorted(indexer)[0])
        divisions.append(sorted(items[-1][1])[-1])
        graph = HighLevelGraph.from_collections(name, dsk, dependencies=[self.obj])
    else:
        divisions = [None, None]
        dsk = {(name, 0): meta.head(0)}
        graph = HighLevelGraph.from_collections(name, dsk)
    return new_dd_object(graph, name, meta=meta, divisions=divisions)