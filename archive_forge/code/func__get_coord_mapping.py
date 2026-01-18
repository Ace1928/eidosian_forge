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
def _get_coord_mapping(dims, output, out_indices, numblocks, argpairs, concatenate):
    """Calculate coordinate mapping for graph construction.

    This function handles the high-level logic behind Blockwise graph
    construction. The output is a tuple containing: The mapping between
    input and output block coordinates (`coord_maps`), the axes along
    which to concatenate for each input (`concat_axes`), and the dummy
    indices needed for broadcasting (`dummies`).

    Used by `make_blockwise_graph` and `Blockwise._cull_dependencies`.

    Parameters
    ----------
    dims : dict
        Mapping between each index specified in `argpairs` and
        the number of output blocks for that index. Corresponds
        to the Blockwise `dims` attribute.
    output : str
        Corresponds to the Blockwise `output` attribute.
    out_indices : tuple
        Corresponds to the Blockwise `output_indices` attribute.
    numblocks : dict
        Corresponds to the Blockwise `numblocks` attribute.
    argpairs : tuple
        Corresponds to the Blockwise `indices` attribute.
    concatenate : bool
        Corresponds to the Blockwise `concatenate` attribute.
    """
    block_names = set()
    all_indices = set()
    for name, ind in argpairs:
        if ind is not None:
            block_names.add(name)
            for x in ind:
                all_indices.add(x)
    assert set(numblocks) == block_names
    dummy_indices = all_indices - set(out_indices)
    index_pos, zero_pos = ({}, {})
    for i, ind in enumerate(out_indices):
        index_pos[ind] = i
        zero_pos[ind] = -1
    _dummies_list = []
    for i, ind in enumerate(dummy_indices):
        index_pos[ind] = 2 * i + len(out_indices)
        zero_pos[ind] = 2 * i + 1 + len(out_indices)
        reps = 1 if concatenate else dims[ind]
        _dummies_list.append([list(range(dims[ind])), [0] * reps])
    dummies = tuple(itertools.chain.from_iterable(_dummies_list))
    dummies += (0,)
    coord_maps = []
    concat_axes = []
    for arg, ind in argpairs:
        if ind is not None:
            coord_maps.append([zero_pos[i] if nb == 1 else index_pos[i] for i, nb in zip(ind, numblocks[arg])])
            concat_axes.append([n for n, i in enumerate(ind) if i in dummy_indices])
        else:
            coord_maps.append(None)
            concat_axes.append(None)
    return (coord_maps, concat_axes, dummies)