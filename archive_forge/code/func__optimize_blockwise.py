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
def _optimize_blockwise(full_graph, keys=()):
    keep = {k[0] if type(k) is tuple else k for k in keys}
    layers = full_graph.layers
    dependents = reverse_dict(full_graph.dependencies)
    roots = {k for k in full_graph.layers if not dependents.get(k)}
    stack = list(roots)
    out = {}
    dependencies = {}
    seen = set()
    io_names = set()
    while stack:
        layer = stack.pop()
        if layer in seen or layer not in layers:
            continue
        seen.add(layer)
        if isinstance(layers[layer], Blockwise):
            blockwise_layers = {layer}
            deps = set(blockwise_layers)
            io_names |= layers[layer].io_deps.keys()
            while deps:
                dep = deps.pop()
                if dep not in layers:
                    stack.append(dep)
                    continue
                if not isinstance(layers[dep], Blockwise):
                    stack.append(dep)
                    continue
                if dep != layer and dep in keep:
                    stack.append(dep)
                    continue
                if layers[dep].concatenate != layers[layer].concatenate:
                    stack.append(dep)
                    continue
                if sum((k == dep for k, ind in layers[layer].indices if ind is not None)) > 1:
                    stack.append(dep)
                    continue
                if blockwise_layers and (not _can_fuse_annotations(layers[next(iter(blockwise_layers))].annotations, layers[dep].annotations)):
                    stack.append(dep)
                    continue
                blockwise_layers.add(dep)
                output_indices = set(layers[dep].output_indices)
                input_indices = {i for _, ind in layers[dep].indices if ind for i in ind}
                is_io_superset = output_indices.issuperset(input_indices)
                for d in full_graph.dependencies.get(dep, ()):
                    if is_io_superset and len(dependents[d]) <= 1:
                        deps.add(d)
                    else:
                        stack.append(d)
            new_layer = rewrite_blockwise([layers[l] for l in blockwise_layers])
            out[layer] = new_layer
            new_deps = set()
            for l in blockwise_layers:
                new_deps |= set({d for d in full_graph.dependencies[l] if d not in blockwise_layers and d in full_graph.dependencies})
            for k, v in new_layer.indices:
                if v is None:
                    new_deps |= keys_in_tasks(full_graph.dependencies, [k])
                elif k not in io_names:
                    new_deps.add(k)
            dependencies[layer] = new_deps
        else:
            out[layer] = layers[layer]
            dependencies[layer] = full_graph.dependencies.get(layer, set())
            stack.extend(full_graph.dependencies.get(layer, ()))
    return HighLevelGraph(out, dependencies)