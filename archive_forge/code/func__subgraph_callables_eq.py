from __future__ import annotations
import itertools
import pickle
from functools import partial
import pytest
import dask
from dask.base import tokenize
from dask.core import get_dependencies
from dask.local import get_sync
from dask.optimization import (
from dask.utils import apply, partial_by_order
from dask.utils_test import add, inc
def _subgraph_callables_eq(self, other):
    return type(self) is type(other) and self.outkey == other.outkey and (set(self.inkeys) == set(other.inkeys)) and (tokenize(self.dsk) == tokenize(other.dsk))