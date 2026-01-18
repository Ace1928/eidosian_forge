from __future__ import annotations
import functools
import math
import operator
from collections import defaultdict
from collections.abc import Callable
from itertools import product
from typing import Any
import tlz as toolz
from tlz.curried import map
from dask.base import tokenize
from dask.blockwise import Blockwise, BlockwiseDep, BlockwiseDepDict, blockwise_token
from dask.core import flatten
from dask.highlevelgraph import Layer
from dask.utils import apply, cached_cumsum, concrete, insert
def _keys_to_parts(self, keys):
    """Simple utility to convert keys to partition indices."""
    parts = set()
    for key in keys:
        try:
            _name, _part = key
        except ValueError:
            continue
        if _name != self.name:
            continue
        parts.add(_part)
    return parts