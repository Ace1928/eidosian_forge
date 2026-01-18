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
class ArraySliceDep(ArrayBlockwiseDep):
    """Produce slice(s) into the full-sized array given a chunk index"""
    starts: tuple[tuple[int, ...], ...]

    def __init__(self, chunks: tuple[tuple[int, ...], ...]):
        super().__init__(chunks)
        self.starts = tuple((cached_cumsum(c, initial_zero=True) for c in chunks))

    def __getitem__(self, idx: tuple):
        loc = tuple(((start[i], start[i + 1]) for i, start in zip(idx, self.starts)))
        return tuple((slice(*s, None) for s in loc))