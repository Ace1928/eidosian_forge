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
def _can_fuse_annotations(a: dict | None, b: dict | None) -> bool:
    """
    Treat the special annotation keys, as fusable since we can apply simple
    rules to capture their intent in a fused layer.
    """
    if a == b:
        return True
    if dask.config.get('optimization.annotations.fuse') is False:
        return False
    fusable = {'retries', 'priority', 'resources', 'workers', 'allow_other_workers'}
    if (not a or all((k in fusable for k in a))) and (not b or all((k in fusable for k in b))):
        return True
    return False