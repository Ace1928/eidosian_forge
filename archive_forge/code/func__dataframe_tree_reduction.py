from __future__ import annotations
import os
import pytest
import sys
from operator import getitem
from distributed import Client, SchedulerPlugin
from distributed.utils_test import cluster, loop  # noqa F401
from dask.highlevelgraph import HighLevelGraph
from dask.layers import ArrayChunkShapeDep, ArraySliceDep, fractional_slice
def _dataframe_tree_reduction(tmpdir):
    pd = pytest.importorskip('pandas')
    dd = pytest.importorskip('dask.dataframe')
    df = pd.DataFrame({'a': range(10), 'b': range(10, 20)})
    return dd.from_pandas(df, npartitions=2).mean()