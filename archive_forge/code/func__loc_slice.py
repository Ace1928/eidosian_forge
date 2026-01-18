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
def _loc_slice(self, iindexer, cindexer):
    name = 'loc-%s' % tokenize(iindexer, cindexer, self)
    assert isinstance(iindexer, slice)
    assert iindexer.step in (None, 1)
    if iindexer.start is not None:
        start = self._get_partitions(iindexer.start)
    else:
        start = 0
    if iindexer.stop is not None:
        stop = self._get_partitions(iindexer.stop)
    else:
        stop = self.obj.npartitions - 1
    if iindexer.start is None and self.obj.known_divisions:
        istart = self.obj.divisions[0] if iindexer.stop is None else min(self.obj.divisions[0], iindexer.stop)
    else:
        istart = self._coerce_loc_index(iindexer.start)
    if iindexer.stop is None and self.obj.known_divisions:
        istop = self.obj.divisions[-1] if iindexer.start is None else max(self.obj.divisions[-1], iindexer.start)
    else:
        istop = self._coerce_loc_index(iindexer.stop)
    if stop == start:
        dsk = {(name, 0): (methods.loc, (self._name, start), slice(iindexer.start, iindexer.stop), cindexer)}
        divisions = [istart, istop]
    else:
        dsk = {(name, 0): (methods.loc, (self._name, start), slice(iindexer.start, None), cindexer)}
        for i in range(1, stop - start):
            if cindexer is None:
                dsk[name, i] = (self._name, start + i)
            else:
                dsk[name, i] = (methods.loc, (self._name, start + i), slice(None, None), cindexer)
        dsk[name, stop - start] = (methods.loc, (self._name, stop), slice(None, iindexer.stop), cindexer)
        if iindexer.start is None:
            div_start = self.obj.divisions[0]
        else:
            div_start = max(istart, self.obj.divisions[start])
        if iindexer.stop is None:
            div_stop = self.obj.divisions[-1]
        else:
            div_stop = min(istop, self.obj.divisions[stop + 1])
        divisions = (div_start,) + self.obj.divisions[start + 1:stop + 1] + (div_stop,)
    assert len(divisions) == len(dsk) + 1
    meta = self._make_meta(iindexer, cindexer)
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[self.obj])
    return new_dd_object(graph, name, meta=meta, divisions=divisions)