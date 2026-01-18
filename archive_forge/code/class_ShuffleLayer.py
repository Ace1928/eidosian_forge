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
class ShuffleLayer(SimpleShuffleLayer):
    """Shuffle-stage HighLevelGraph layer

    High-level graph layer corresponding to a single stage of
    a multi-stage inter-partition shuffle operation.

    Stage: (shuffle-group) -> (shuffle-split) -> (shuffle-join)

    Parameters
    ----------
    name : str
        Name of new (partially) shuffled collection.
    column : str or list of str
        Column(s) to be used to map rows to output partitions (by hashing).
    inputs : list of tuples
        Each tuple dictates the data movement for a specific partition.
    stage : int
        Index of the current shuffle stage.
    npartitions : int
        Number of output partitions for the full (multi-stage) shuffle.
    npartitions_input : int
        Number of partitions in the original (un-shuffled) DataFrame.
    k : int
        A partition is split into this many groups during each stage.
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

    def __init__(self, name, column, inputs, stage, npartitions, npartitions_input, nsplits, ignore_index, name_input, meta_input, parts_out=None, annotations=None):
        self.inputs = inputs
        self.stage = stage
        self.nsplits = nsplits
        super().__init__(name, column, npartitions, npartitions_input, ignore_index, name_input, meta_input, parts_out=parts_out or range(len(inputs)), annotations=annotations)

    def __repr__(self):
        return "ShuffleLayer<name='{}', stage={}, nsplits={}, npartitions={}>".format(self.name, self.stage, self.nsplits, self.npartitions)

    def _cull_dependencies(self, keys, parts_out=None):
        """Determine the necessary dependencies to produce `keys`.

        Does not require graph materialization.
        """
        deps = defaultdict(set)
        parts_out = parts_out or self._keys_to_parts(keys)
        inp_part_map = {inp: i for i, inp in enumerate(self.inputs)}
        for part in parts_out:
            out = self.inputs[part]
            for k in range(self.nsplits):
                _inp = insert(out, self.stage, k)
                _part = inp_part_map[_inp]
                if self.stage == 0 and _part >= self.npartitions_input:
                    deps[self.name, part].add(('group-' + self.name, _inp, 'empty'))
                else:
                    deps[self.name, part].add((self.name_input, _part))
        return deps

    def _cull(self, parts_out):
        return ShuffleLayer(self.name, self.column, self.inputs, self.stage, self.npartitions, self.npartitions_input, self.nsplits, self.ignore_index, self.name_input, self.meta_input, parts_out=parts_out)

    def _construct_graph(self, deserializing=False):
        """Construct graph for a "rearrange-by-column" stage."""
        shuffle_group_name = 'group-' + self.name
        if deserializing:
            concat_func = CallableLazyImport('dask.dataframe.core._concat')
            shuffle_group_func = CallableLazyImport('dask.dataframe.shuffle.shuffle_group')
        else:
            from dask.dataframe.core import _concat as concat_func
            from dask.dataframe.shuffle import shuffle_group as shuffle_group_func
        dsk = {}
        inp_part_map = {inp: i for i, inp in enumerate(self.inputs)}
        for part in self.parts_out:
            out = self.inputs[part]
            _concat_list = []
            for i in range(self.nsplits):
                _inp = insert(out, self.stage, i)
                _idx = out[self.stage]
                _concat_list.append((self.split_name, _idx, _inp))
            dsk[self.name, part] = (concat_func, _concat_list, self.ignore_index)
            for _, _idx, _inp in _concat_list:
                dsk[self.split_name, _idx, _inp] = (operator.getitem, (shuffle_group_name, _inp), _idx)
                if (shuffle_group_name, _inp) not in dsk:
                    _part = inp_part_map[_inp]
                    if self.stage == 0:
                        if _part < self.npartitions_input:
                            input_key = (self.name_input, _part)
                        else:
                            input_key = (shuffle_group_name, _inp, 'empty')
                            dsk[input_key] = self.meta_input
                    else:
                        input_key = (self.name_input, _part)
                    dsk[shuffle_group_name, _inp] = (shuffle_group_func, input_key, self.column, self.stage, self.nsplits, self.npartitions_input, self.ignore_index, self.npartitions)
        return dsk