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
class SimpleShuffleLayer(Layer):
    """Simple HighLevelGraph Shuffle layer

    High-level graph layer for a simple shuffle operation in which
    each output partition depends on all input partitions.

    Parameters
    ----------
    name : str
        Name of new shuffled output collection.
    column : str or list of str
        Column(s) to be used to map rows to output partitions (by hashing).
    npartitions : int
        Number of output partitions.
    npartitions_input : int
        Number of partitions in the original (un-shuffled) DataFrame.
    ignore_index: bool, default False
        Ignore index during shuffle.  If ``True``, performance may improve,
        but index values will not be preserved.
    name_input : str
        Name of input collection.
    meta_input : pd.DataFrame-like object
        Empty metadata of input collection.
    parts_out : list of int (optional)
        List of required output-partition indices.
    annotations : dict (optional)
        Layer annotations
    """

    def __init__(self, name, column, npartitions, npartitions_input, ignore_index, name_input, meta_input, parts_out=None, annotations=None):
        self.name = name
        self.column = column
        self.npartitions = npartitions
        self.npartitions_input = npartitions_input
        self.ignore_index = ignore_index
        self.name_input = name_input
        self.meta_input = meta_input
        self.parts_out = parts_out or range(npartitions)
        self.split_name = 'split-' + self.name
        annotations = annotations or {}
        self._split_keys = None
        if 'priority' not in annotations:
            annotations['priority'] = self._key_priority
        super().__init__(annotations=annotations)

    def _key_priority(self, key):
        assert isinstance(key, tuple)
        if key[0] == self.split_name:
            return 1
        else:
            return 0

    def get_output_keys(self):
        return {(self.name, part) for part in self.parts_out}

    def __repr__(self):
        return "SimpleShuffleLayer<name='{}', npartitions={}>".format(self.name, self.npartitions)

    def is_materialized(self):
        return hasattr(self, '_cached_dict')

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

    def _keys_to_parts(self, keys):
        """Simple utility to convert keys to partition indices."""
        parts = set()
        for key in keys:
            try:
                _name, _part = key
            except ValueError:
                continue
            if _name != self.name:
                continue
            parts.add(_part)
        return parts

    def _cull_dependencies(self, keys, parts_out=None):
        """Determine the necessary dependencies to produce `keys`.

        For a simple shuffle, output partitions always depend on
        all input partitions. This method does not require graph
        materialization.
        """
        deps = defaultdict(set)
        parts_out = parts_out or self._keys_to_parts(keys)
        for part in parts_out:
            deps[self.name, part] |= {(self.name_input, i) for i in range(self.npartitions_input)}
        return deps

    def _cull(self, parts_out):
        return SimpleShuffleLayer(self.name, self.column, self.npartitions, self.npartitions_input, self.ignore_index, self.name_input, self.meta_input, parts_out=parts_out)

    def cull(self, keys, all_keys):
        """Cull a SimpleShuffleLayer HighLevelGraph layer.

        The underlying graph will only include the necessary
        tasks to produce the keys (indices) included in `parts_out`.
        Therefore, "culling" the layer only requires us to reset this
        parameter.
        """
        parts_out = self._keys_to_parts(keys)
        culled_deps = self._cull_dependencies(keys, parts_out=parts_out)
        if parts_out != set(self.parts_out):
            culled_layer = self._cull(parts_out)
            return (culled_layer, culled_deps)
        else:
            return (self, culled_deps)

    def _construct_graph(self, deserializing=False):
        """Construct graph for a simple shuffle operation."""
        shuffle_group_name = 'group-' + self.name
        if deserializing:
            concat_func = CallableLazyImport('dask.dataframe.core._concat')
            shuffle_group_func = CallableLazyImport('dask.dataframe.shuffle.shuffle_group')
        else:
            from dask.dataframe.core import _concat as concat_func
            from dask.dataframe.shuffle import shuffle_group as shuffle_group_func
        dsk = {}
        for part_out in self.parts_out:
            _concat_list = [(self.split_name, part_out, part_in) for part_in in range(self.npartitions_input)]
            dsk[self.name, part_out] = (concat_func, _concat_list, self.ignore_index)
            for _, _part_out, _part_in in _concat_list:
                dsk[self.split_name, _part_out, _part_in] = (operator.getitem, (shuffle_group_name, _part_in), _part_out)
                if (shuffle_group_name, _part_in) not in dsk:
                    dsk[shuffle_group_name, _part_in] = (shuffle_group_func, (self.name_input, _part_in), self.column, 0, self.npartitions, self.npartitions, self.ignore_index, self.npartitions)
        return dsk