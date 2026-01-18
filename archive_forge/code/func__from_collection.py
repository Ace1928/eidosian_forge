from __future__ import annotations
import abc
import copy
import html
from collections.abc import (
from typing import Any
import tlz as toolz
import dask
from dask import config
from dask.base import clone_key, flatten, is_dask_collection, normalize_token
from dask.core import keys_in_tasks, reverse_dict
from dask.typing import DaskCollection, Graph, Key
from dask.utils import ensure_dict, import_required, key_split
from dask.widgets import get_template
@classmethod
def _from_collection(cls, name, layer, collection):
    """`from_collections` optimized for a single collection"""
    if not is_dask_collection(collection):
        raise TypeError(type(collection))
    graph = collection.__dask_graph__()
    if isinstance(graph, HighLevelGraph):
        layers = ensure_dict(graph.layers, copy=True)
        layers[name] = layer
        deps = ensure_dict(graph.dependencies, copy=True)
        deps[name] = set(collection.__dask_layers__())
    else:
        key = _get_some_layer_name(collection)
        layers = {name: layer, key: graph}
        deps = {name: {key}, key: set()}
    return cls(layers, deps)