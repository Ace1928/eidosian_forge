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
class Blockwise(Layer):
    """Tensor Operation

    This is a lazily constructed mapping for tensor operation graphs.
    This defines a dictionary using an operation and an indexing pattern.
    It is built for many operations like elementwise, transpose, tensordot, and
    so on.  We choose to keep these as symbolic mappings rather than raw
    dictionaries because we are able to fuse them during optimization,
    sometimes resulting in much lower overhead.

    Parameters
    ----------
    output: str
        The name of the output collection.  Used in keynames
    output_indices: tuple
        The output indices, like ``('i', 'j', 'k')`` used to determine the
        structure of the block computations
    dsk: dict
        A small graph to apply per-output-block.  May include keys from the
        input indices.
    indices: tuple[tuple[str, tuple[str, ...] | None], ...]
        An ordered mapping from input key name, like ``'x'``
        to input indices, like ``('i', 'j')``
        Or includes literals, which have ``None`` for an index value.
        In place of input-key names, the first tuple element may also be a
        ``BlockwiseDep`` object.
    numblocks: Mapping[key, Sequence[int]]
        Number of blocks along each dimension for each input
    concatenate: bool
        Whether or not to pass contracted dimensions as a list of inputs or a
        single input to the block function
    new_axes: Mapping
        New index dimensions that may have been created and their size,
        e.g. ``{'j': 2, 'k': 3}``
    output_blocks: set[tuple[int, ...]]
        Specify a specific set of required output blocks. Since the graph
        will only contain the necessary tasks to generate these outputs,
        this kwarg can be used to "cull" the abstract layer (without needing
        to materialize the low-level graph).
    annotations: dict (optional)
        Layer annotations
    io_deps: dict[str, BlockwiseDep] (optional)
        Dictionary containing the mapping between "place-holder" collection
        keys and ``BlockwiseDep``-based objects.
        **WARNING**: This argument should only be used internally (for culling,
        fusion and cloning of existing Blockwise layers). Explicit use of this
        argument will be deprecated in the future.

    See Also
    --------
    dask.blockwise.blockwise
    dask.array.blockwise
    """
    output: str
    output_indices: tuple[str, ...]
    dsk: Graph
    indices: tuple[tuple[str, tuple[str, ...] | None], ...]
    numblocks: Mapping[str, Sequence[int]]
    concatenate: bool | None
    new_axes: Mapping[str, int]
    output_blocks: set[tuple[int, ...]] | None
    io_deps: Mapping[str, BlockwiseDep]

    def __init__(self, output: str, output_indices: Iterable[str], dsk: Graph, indices: Iterable[tuple[str | BlockwiseDep, Iterable[str] | None]], numblocks: Mapping[str, Sequence[int]], concatenate: bool | None=None, new_axes: Mapping[str, int] | None=None, output_blocks: set[tuple[int, ...]] | None=None, annotations: Mapping[str, Any] | None=None, io_deps: Mapping[str, BlockwiseDep] | None=None):
        super().__init__(annotations=annotations)
        self.output = output
        self.output_indices = tuple(output_indices)
        self.output_blocks = output_blocks
        self.dsk = dsk
        _tmp_indices = []
        if indices:
            numblocks = ensure_dict(numblocks, copy=True)
            io_deps = ensure_dict(io_deps or {}, copy=True)
            for dep, ind in indices:
                if isinstance(dep, BlockwiseDep):
                    name = tokenize(dep)
                    io_deps[name] = dep
                    numblocks[name] = dep.numblocks
                else:
                    name = dep
                _tmp_indices.append((name, tuple(ind) if ind is not None else ind))
        self.numblocks = numblocks
        self.io_deps = io_deps or {}
        self.indices = tuple(_tmp_indices)
        output_indices_set = set(self.output_indices)
        if concatenate is not None and all((i in output_indices_set for name, ind in self.indices if ind is not None for i in ind)):
            concatenate = None
        self.concatenate = concatenate
        self.new_axes = new_axes or {}

    @property
    def dims(self):
        """Returns a dictionary mapping between each index specified in
        `self.indices` and the number of output blocks for that indice.
        """
        if not hasattr(self, '_dims'):
            self._dims = _make_dims(self.indices, self.numblocks, self.new_axes)
        return self._dims

    def __repr__(self):
        return f'Blockwise<{self.indices} -> {self.output}>'

    @property
    def _dict(self):
        if hasattr(self, '_cached_dict'):
            return self._cached_dict['dsk']
        else:
            keys = tuple(map(blockwise_token, range(len(self.indices))))
            dsk, _ = fuse(self.dsk, [self.output])
            func = SubgraphCallable(dsk, self.output, keys)
            dsk = make_blockwise_graph(func, self.output, self.output_indices, *list(toolz.concat(self.indices)), new_axes=self.new_axes, numblocks=self.numblocks, concatenate=self.concatenate, output_blocks=self.output_blocks, dims=self.dims, io_deps=self.io_deps)
            self._cached_dict = {'dsk': dsk}
        return self._cached_dict['dsk']

    def get_output_keys(self):
        if self.output_blocks:
            return {(self.output, *p) for p in self.output_blocks}
        return {(self.output, *p) for p in itertools.product(*[range(self.dims[i]) for i in self.output_indices])}

    def __getitem__(self, key):
        return self._dict[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self.output_blocks) if self.output_blocks else prod((self.dims[i] for i in self.output_indices))

    def is_materialized(self):
        return hasattr(self, '_cached_dict')

    def _cull_dependencies(self, all_hlg_keys, output_blocks):
        """Determine the necessary dependencies to produce `output_blocks`.

        This method does not require graph materialization.
        """
        concatenate = None
        if self.concatenate is True:
            from dask.array.core import concatenate_axes as concatenate
        coord_maps, concat_axes, dummies = _get_coord_mapping(self.dims, self.output, self.output_indices, self.numblocks, self.indices, concatenate)
        const_deps = set()
        for arg, ind in self.indices:
            if ind is None:
                try:
                    if arg in all_hlg_keys:
                        const_deps.add(arg)
                except TypeError:
                    pass
        key_deps = {}
        for out_coords in output_blocks:
            deps = set()
            coords = out_coords + dummies
            for cmap, axes, (arg, ind) in zip(coord_maps, concat_axes, self.indices):
                if ind is not None and arg not in self.io_deps:
                    arg_coords = tuple((coords[c] for c in cmap))
                    if axes:
                        tups = lol_product((arg,), arg_coords)
                        deps.update(flatten(tups))
                        if concatenate:
                            tups = (concatenate, tups, axes)
                    else:
                        tups = (arg,) + arg_coords
                        deps.add(tups)
            key_deps[(self.output,) + out_coords] = deps | const_deps
        for key, io_dep in self.io_deps.items():
            if io_dep.produces_keys:
                for out_coords in output_blocks:
                    key = (self.output,) + out_coords
                    valid_key_dep = io_dep[out_coords]
                    key_deps[key] |= {valid_key_dep}
        return key_deps

    def _cull(self, output_blocks):
        return Blockwise(self.output, self.output_indices, self.dsk, self.indices, self.numblocks, concatenate=self.concatenate, new_axes=self.new_axes, output_blocks=output_blocks, annotations=self.annotations, io_deps=self.io_deps)

    def cull(self, keys: set, all_hlg_keys: Iterable) -> tuple[Layer, Mapping[Key, set[Key]]]:
        output_blocks: set[tuple[int, ...]] = set()
        for key in keys:
            if key[0] == self.output:
                output_blocks.add(key[1:])
        culled_deps = self._cull_dependencies(all_hlg_keys, output_blocks)
        out_size_iter = (self.dims[i] for i in self.output_indices)
        if prod(out_size_iter) != len(culled_deps):
            culled_layer = self._cull(output_blocks)
            return (culled_layer, culled_deps)
        else:
            return (self, culled_deps)

    def clone(self, keys: set[Key], seed: Hashable, bind_to: Key | None=None) -> tuple[Layer, bool]:
        names = {get_name_from_key(k) for k in keys}
        if 'PYTEST_CURRENT_TEST' in os.environ:
            assert not self.get_output_keys() - keys
            for name, nb in self.numblocks.items():
                if name in names:
                    for block in product(*(list(range(nbi)) for nbi in nb)):
                        assert (name, *block) in keys
        is_leaf = True
        indices = []
        k: Key
        for k, idxv in self.indices:
            if ishashable(k) and k in names:
                is_leaf = False
                k = clone_key(k, seed)
            indices.append((k, idxv))
        numblocks: dict[str, Sequence[int]] = {}
        for k, nbv in self.numblocks.items():
            if k in names:
                is_leaf = False
                k = clone_key(k, seed)
            numblocks[k] = nbv
        dsk = {clone_key(k, seed): v for k, v in self.dsk.items()}
        if bind_to is not None and is_leaf:
            from dask.graph_manipulation import chunks
            assert isinstance(bind_to, str)
            dsk = {k: (chunks.bind, v, f'_{len(indices)}') for k, v in dsk.items()}
            indices.append((bind_to, None))
        return (Blockwise(output=clone_key(self.output, seed), output_indices=self.output_indices, dsk=dsk, indices=indices, numblocks=numblocks, concatenate=self.concatenate, new_axes=self.new_axes, output_blocks=self.output_blocks, annotations=self.annotations, io_deps=self.io_deps), bind_to is not None and is_leaf)