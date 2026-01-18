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
class BlockwiseDepDict(BlockwiseDep):
    """Dictionary-based Blockwise-IO argument

    This is a dictionary-backed instance of ``BlockwiseDep``.
    The purpose of this class is to simplify the construction
    of IO-based Blockwise Layers with block/partition-dependent
    function arguments that are difficult to calculate at
    graph-materialization time.

    Examples
    --------

    Specify an IO-based function for the Blockwise Layer. Note
    that the function will be passed a single input object when
    the task is executed (e.g. a single ``tuple`` or ``dict``):

    >>> import pandas as pd
    >>> func = lambda x: pd.read_csv(**x)

    Use ``BlockwiseDepDict`` to define the input argument to
    ``func`` for each block/partition:

    >>> dep = BlockwiseDepDict(
    ...     mapping={
    ...         (0,) : {
    ...             "filepath_or_buffer": "data.csv",
    ...             "skiprows": 1,
    ...             "nrows": 2,
    ...             "names": ["a", "b"],
    ...         },
    ...         (1,) : {
    ...             "filepath_or_buffer": "data.csv",
    ...             "skiprows": 3,
    ...             "nrows": 2,
    ...             "names": ["a", "b"],
    ...         },
    ...     }
    ... )

    Construct a Blockwise Layer with ``dep`` specified
    in the ``indices`` list:

    >>> layer = Blockwise(
    ...     output="collection-name",
    ...     output_indices="i",
    ...     dsk={"collection-name": (func, '_0')},
    ...     indices=[(dep, "i")],
    ...     numblocks={},
    ... )

    See Also
    --------
    dask.blockwise.Blockwise
    dask.blockwise.BlockwiseDep
    """

    def __init__(self, mapping: dict, numblocks: tuple[int, ...] | None=None, produces_tasks: bool=False, produces_keys: bool=False):
        self.mapping = mapping
        self.produces_tasks = produces_tasks
        self.numblocks = numblocks or (len(mapping),)
        self._produces_keys = produces_keys

    @property
    def produces_keys(self) -> bool:
        return self._produces_keys

    def __getitem__(self, idx: tuple[int, ...]) -> Any:
        try:
            return self.mapping[idx]
        except KeyError as err:
            flat_idx = idx[:len(self.numblocks)]
            if flat_idx in self.mapping:
                return self.mapping[flat_idx]
            raise err

    def __len__(self) -> int:
        return len(self.mapping)