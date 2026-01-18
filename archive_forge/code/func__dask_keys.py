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
def _dask_keys(self):
    if self._cached_keys is not None:
        return self._cached_keys
    name, chunks, numblocks = (self.name, self.chunks, self.numblocks)

    def keys(*args):
        if not chunks:
            return [(name,)]
        ind = len(args)
        if ind + 1 == len(numblocks):
            result = [(name,) + args + (i,) for i in range(numblocks[ind])]
        else:
            result = [keys(*args + (i,)) for i in range(numblocks[ind])]
        return result
    self._cached_keys = result = keys()
    return result