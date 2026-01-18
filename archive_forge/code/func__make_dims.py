from __future__ import annotations
import itertools
import os
from collections.abc import Hashable, Iterable, Mapping, Sequence
from itertools import product
from math import prod
from typing import Any
import tlz as toolz
import dask
from dask.base import clone_key, get_name_from_key, tokenize
from dask.core import flatten, ishashable, keys_in_tasks, reverse_dict
from dask.highlevelgraph import HighLevelGraph, Layer
from dask.optimization import SubgraphCallable, fuse
from dask.typing import Graph, Key
from dask.utils import (
def _make_dims(indices, numblocks, new_axes):
    """Returns a dictionary mapping between each index specified in
    `indices` and the number of output blocks for that indice.
    """
    dims = broadcast_dimensions(indices, numblocks)
    for k, v in new_axes.items():
        dims[k] = len(v) if isinstance(v, tuple) else 1
    return dims