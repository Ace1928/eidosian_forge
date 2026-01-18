from __future__ import annotations
import math
import numbers
from collections.abc import Iterable
from enum import Enum
from typing import Any
from dask import config, core, utils
from dask.base import normalize_token, tokenize
from dask.core import (
from dask.typing import Graph, Key
def _inplace_fuse_subgraphs(dsk, keys, dependencies, fused_trees, rename_keys):
    """Subroutine of fuse.

    Mutates dsk, dependencies, and fused_trees inplace"""
    child2parent = {}
    unfusible = set()
    for parent in dsk:
        deps = dependencies[parent]
        has_many_children = len(deps) > 1
        for child in deps:
            if keys is not None and child in keys:
                unfusible.add(child)
            elif child in child2parent:
                del child2parent[child]
                unfusible.add(child)
            elif has_many_children:
                unfusible.add(child)
            elif child not in unfusible:
                child2parent[child] = parent
    chains = []
    parent2child = {v: k for k, v in child2parent.items()}
    while child2parent:
        child, parent = child2parent.popitem()
        chain = [child, parent]
        while parent in child2parent:
            parent = child2parent.pop(parent)
            del parent2child[parent]
            chain.append(parent)
        chain.reverse()
        while child in parent2child:
            child = parent2child.pop(child)
            del child2parent[child]
            chain.append(child)
        ntasks = 0
        for key in chain:
            ntasks += istask(dsk[key])
            if ntasks > 1:
                chains.append(chain)
                break
    for chain in chains:
        subgraph = {k: dsk[k] for k in chain}
        outkey = chain[0]
        inkeys_set = dependencies[outkey] = dependencies[chain[-1]]
        for k in chain[1:]:
            del dependencies[k]
            del dsk[k]
        inkeys = tuple(inkeys_set)
        dsk[outkey] = (SubgraphCallable(subgraph, outkey, inkeys),) + inkeys
        if rename_keys:
            chain2 = []
            for k in chain:
                subchain = fused_trees.pop(k, False)
                if subchain:
                    chain2.extend(subchain)
                else:
                    chain2.append(k)
            fused_trees[outkey] = chain2