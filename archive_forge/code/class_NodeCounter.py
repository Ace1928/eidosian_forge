from __future__ import annotations
import random
import time
from operator import add
import pytest
import dask
import dask.bag as db
from dask import delayed
from dask.base import clone_key
from dask.blockwise import Blockwise
from dask.graph_manipulation import bind, checkpoint, chunks, clone, wait_on
from dask.highlevelgraph import HighLevelGraph
from dask.tests.test_base import Tuple
from dask.utils_test import import_or_none
class NodeCounter:

    def __init__(self):
        self.n = 0

    def __dask_tokenize__(self):
        return (type(self), self.n)

    def f(self, x):
        time.sleep(random.random() / 100)
        self.n += 1
        return x