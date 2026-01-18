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
def _cull_dependencies(self, keys, parts_out=None):
    """Determine the necessary dependencies to produce `keys`.

        For a broadcast join, output partitions always depend on
        all partitions of the broadcasted collection, but only one
        partition of the "other" collection.
        """
    bcast_name, bcast_size, other_name = self._broadcast_plan[:3]
    deps = defaultdict(set)
    parts_out = parts_out or self._keys_to_parts(keys)
    for part in parts_out:
        deps[self.name, part] |= {(bcast_name, i) for i in range(bcast_size)}
        deps[self.name, part] |= {(other_name, part)}
    return deps