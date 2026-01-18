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
class NonHashableCallable:

    def __call__(self, a):
        return a + 1

    def __hash__(self):
        raise TypeError('Not hashable')