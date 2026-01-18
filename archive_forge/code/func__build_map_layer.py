from __future__ import annotations
import uuid
from collections.abc import Callable, Hashable
from typing import Literal, TypeVar
from dask.base import (
from dask.blockwise import blockwise
from dask.core import flatten
from dask.delayed import Delayed, delayed
from dask.highlevelgraph import HighLevelGraph, Layer, MaterializedLayer
from dask.typing import Graph, Key
def _build_map_layer(func: Callable, prev_name: str, new_name: str, collection, dependencies: tuple[Delayed, ...]=()) -> Layer:
    """Apply func to all keys of collection. Create a Blockwise layer whenever possible;
    fall back to MaterializedLayer otherwise.

    Parameters
    ----------
    func
        Callable to be invoked on the graph node
    prev_name : str
        name of the layer to map from; in case of dask base collections, this is the
        collection name. Note how third-party collections, e.g. xarray.Dataset, can
        have multiple names.
    new_name : str
        name of the layer to map to
    collection
        Arbitrary dask collection
    dependencies
        Zero or more Delayed objects, which will be passed as arbitrary variadic args to
        func after the collection's chunk
    """
    if _can_apply_blockwise(collection):
        try:
            numblocks = collection.numblocks
        except AttributeError:
            numblocks = (collection.npartitions,)
        indices = tuple((i for i, _ in enumerate(numblocks)))
        kwargs = {'_deps': [d.key for d in dependencies]} if dependencies else {}
        return blockwise(func, new_name, indices, prev_name, indices, numblocks={prev_name: numblocks}, dependencies=dependencies, **kwargs)
    else:
        dep_keys = tuple((d.key for d in dependencies))
        return MaterializedLayer({replace_name_in_key(k, {prev_name: new_name}): (func, k) + dep_keys for k in flatten(collection.__dask_keys__()) if get_name_from_key(k) == prev_name})