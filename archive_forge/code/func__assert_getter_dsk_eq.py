from __future__ import annotations
import pytest
import numpy as np
import dask
import dask.array as da
from dask.array.chunk import getitem as da_getitem
from dask.array.core import getter as da_getter
from dask.array.core import getter_nofancy as da_getter_nofancy
from dask.array.optimization import (
from dask.array.utils import assert_eq
from dask.highlevelgraph import HighLevelGraph
from dask.optimization import SubgraphCallable, fuse
from dask.utils import SerializableLock
def _assert_getter_dsk_eq(a, b):
    """
    Compare two getter dsks.

    TODO: this is here to support the fact that low-level array slice fusion needs to be
    able to introspect slicing tasks. But some slicing tasks (e.g. `from_array`) could
    be hidden within SubgraphCallables. This and _check_get_task_eq should be removed
    when high-level slicing lands, and replaced with basic equality checks.
    """
    assert a.keys() == b.keys()
    for k, av in a.items():
        bv = b[k]
        if dask.core.istask(av):
            assert _check_get_task_eq(av, bv)
        else:
            assert av == bv