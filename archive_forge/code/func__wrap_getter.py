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
def _wrap_getter(func, wrap):
    """
    Getters generated from a Blockwise layer might be wrapped in a SubgraphCallable.
    Make sure that the optimization functions can still work if that is the case.
    """
    if wrap:
        return SubgraphCallable({'key': (func, 'index')}, outkey='key', inkeys='index')
    else:
        return func