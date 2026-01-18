from __future__ import annotations
import contextlib
import importlib
import numbers
from itertools import chain, product
from numbers import Integral
from operator import getitem
from threading import Lock
import numpy as np
from dask.array.backends import array_creation_dispatch
from dask.array.core import (
from dask.array.creation import arange
from dask.array.utils import asarray_safe
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.utils import derived_from, random_state_data, typename
def _rng_from_bitgen(bitgen):
    backend_name = typename(bitgen).split('.')[0]
    backend_lib = importlib.import_module(backend_name)
    return backend_lib.random.default_rng(bitgen)