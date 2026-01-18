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
def _checkpoint_one(collection, split_every) -> Delayed:
    tok = tokenize(collection)
    name = 'checkpoint-' + tok
    keys_iter = flatten(collection.__dask_keys__())
    try:
        next(keys_iter)
        next(keys_iter)
    except StopIteration:
        layer: Graph = {name: (chunks.checkpoint, collection.__dask_keys__())}
        dsk = HighLevelGraph.from_collections(name, layer, dependencies=(collection,))
        return Delayed(name, dsk)
    dsks = []
    map_names = set()
    map_keys = []
    for prev_name in get_collection_names(collection):
        map_name = 'checkpoint_map-' + tokenize(prev_name, tok)
        map_names.add(map_name)
        map_layer = _build_map_layer(chunks.checkpoint, prev_name, map_name, collection)
        map_keys += list(map_layer.get_output_keys())
        dsks.append(HighLevelGraph.from_collections(map_name, map_layer, dependencies=(collection,)))
    reduce_layer: dict = {}
    while split_every and len(map_keys) > split_every:
        k = (name, len(reduce_layer))
        reduce_layer[k] = (chunks.checkpoint, map_keys[:split_every])
        map_keys = map_keys[split_every:] + [k]
    reduce_layer[name] = (chunks.checkpoint, map_keys)
    dsks.append(HighLevelGraph({name: reduce_layer}, dependencies={name: map_names}))
    dsk = HighLevelGraph.merge(*dsks)
    return Delayed(name, dsk)