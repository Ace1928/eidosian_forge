from __future__ import annotations
import os
import pytest
import sys
from operator import getitem
from distributed import Client, SchedulerPlugin
from distributed.utils_test import cluster, loop  # noqa F401
from dask.highlevelgraph import HighLevelGraph
from dask.layers import ArrayChunkShapeDep, ArraySliceDep, fractional_slice
def _pq_fastparquet(tmpdir):
    pytest.importorskip('fastparquet')
    pd = pytest.importorskip('pandas')
    dd = pytest.importorskip('dask.dataframe')
    df = dd.from_pandas(pd.DataFrame({'a': range(10)}), npartitions=2)
    with pytest.warns(FutureWarning):
        df.to_parquet(str(tmpdir), engine='fastparquet')
        return dd.read_parquet(str(tmpdir), engine='fastparquet')