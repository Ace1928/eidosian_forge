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
def _bind_one(child: T, blocker: Delayed | None, omit_layers: set[str], omit_keys: set[Key], seed: Hashable) -> T:
    prev_coll_names = get_collection_names(child)
    if not prev_coll_names:
        return child
    dsk = child.__dask_graph__()
    new_layers: dict[str, Layer] = {}
    new_deps: dict[str, set[str]] = {}
    if isinstance(dsk, HighLevelGraph):
        try:
            layers_to_clone = set(child.__dask_layers__())
        except AttributeError:
            layers_to_clone = prev_coll_names.copy()
    else:
        if len(prev_coll_names) == 1:
            hlg_name = next(iter(prev_coll_names))
        else:
            hlg_name = tokenize(*prev_coll_names)
        dsk = HighLevelGraph.from_collections(hlg_name, dsk)
        layers_to_clone = {hlg_name}
    clone_keys = dsk.get_all_external_keys() - omit_keys
    for layer_name in omit_layers:
        try:
            layer = dsk.layers[layer_name]
        except KeyError:
            continue
        clone_keys -= layer.get_output_keys()
    if blocker is not None:
        blocker_key = blocker.key
        blocker_dsk = blocker.__dask_graph__()
        assert isinstance(blocker_dsk, HighLevelGraph)
        new_layers.update(blocker_dsk.layers)
        new_deps.update(blocker_dsk.dependencies)
    else:
        blocker_key = None
    layers_to_copy_verbatim = set()
    while layers_to_clone:
        prev_layer_name = layers_to_clone.pop()
        new_layer_name = clone_key(prev_layer_name, seed=seed)
        if new_layer_name in new_layers:
            continue
        layer = dsk.layers[prev_layer_name]
        layer_deps = dsk.dependencies[prev_layer_name]
        layer_deps_to_clone = layer_deps - omit_layers
        layer_deps_to_omit = layer_deps & omit_layers
        layers_to_clone |= layer_deps_to_clone
        layers_to_copy_verbatim |= layer_deps_to_omit
        new_layers[new_layer_name], is_bound = layer.clone(keys=clone_keys, seed=seed, bind_to=blocker_key)
        new_dep = {clone_key(dep, seed=seed) for dep in layer_deps_to_clone} | layer_deps_to_omit
        if is_bound:
            new_dep.add(blocker_key)
        new_deps[new_layer_name] = new_dep
    while layers_to_copy_verbatim:
        layer_name = layers_to_copy_verbatim.pop()
        if layer_name in new_layers:
            continue
        layer_deps = dsk.dependencies[layer_name]
        layers_to_copy_verbatim |= layer_deps
        new_deps[layer_name] = layer_deps
        new_layers[layer_name] = dsk.layers[layer_name]
    rebuild, args = child.__dask_postpersist__()
    return rebuild(HighLevelGraph(new_layers, new_deps), *args, rename={prev_name: clone_key(prev_name, seed) for prev_name in prev_coll_names})