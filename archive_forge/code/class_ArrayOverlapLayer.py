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
class ArrayOverlapLayer(Layer):
    """Simple HighLevelGraph array overlap layer.

    Lazily computed High-level graph layer for a array overlap operations.

    Parameters
    ----------
    name : str
        Name of new output overlap array.
    array : Dask array
    axes: Mapping
        Axes dictionary indicating overlap in each dimension,
        e.g. ``{'0': 1, '1': 1}``
    """

    def __init__(self, name, axes, chunks, numblocks, token):
        super().__init__()
        self.name = name
        self.axes = axes
        self.chunks = chunks
        self.numblocks = numblocks
        self.token = token
        self._cached_keys = None

    def __repr__(self):
        return f"ArrayOverlapLayer<name='{self.name}'"

    @property
    def _dict(self):
        """Materialize full dict representation"""
        if hasattr(self, '_cached_dict'):
            return self._cached_dict
        else:
            dsk = self._construct_graph()
            self._cached_dict = dsk
        return self._cached_dict

    def __getitem__(self, key):
        return self._dict[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def is_materialized(self):
        return hasattr(self, '_cached_dict')

    def get_output_keys(self):
        return self.keys()

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

    def _construct_graph(self, deserializing=False):
        """Construct graph for a simple overlap operation."""
        axes = self.axes
        chunks = self.chunks
        name = self.name
        dask_keys = self._dask_keys()
        getitem_name = 'getitem-' + self.token
        overlap_name = 'overlap-' + self.token
        if deserializing:
            concatenate3 = CallableLazyImport('dask.array.core.concatenate3')
        else:
            from dask.array.core import concatenate3
        dims = list(map(len, chunks))
        expand_key2 = functools.partial(_expand_keys_around_center, dims=dims, axes=axes)
        interior_keys = toolz.pipe(dask_keys, flatten, map(expand_key2), map(flatten), toolz.concat, list)
        interior_slices = {}
        overlap_blocks = {}
        for k in interior_keys:
            frac_slice = fractional_slice((name,) + k, axes)
            if (name,) + k != frac_slice:
                interior_slices[(getitem_name,) + k] = frac_slice
            else:
                interior_slices[(getitem_name,) + k] = (name,) + k
                overlap_blocks[(overlap_name,) + k] = (concatenate3, (concrete, expand_key2((None,) + k, name=getitem_name)))
        dsk = toolz.merge(interior_slices, overlap_blocks)
        return dsk