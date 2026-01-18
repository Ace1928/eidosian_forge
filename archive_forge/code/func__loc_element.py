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
def _loc_element(self, iindexer, cindexer):
    name = 'loc-%s' % tokenize(iindexer, self.obj)
    part = self._get_partitions(iindexer)
    if iindexer < self.obj.divisions[0] or iindexer > self.obj.divisions[-1]:
        raise KeyError('the label [%s] is not in the index' % str(iindexer))
    dsk = {(name, 0): (methods.loc, (self._name, part), slice(iindexer, iindexer), cindexer)}
    meta = self._make_meta(iindexer, cindexer)
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[self.obj])
    return new_dd_object(graph, name, meta=meta, divisions=[iindexer, iindexer])