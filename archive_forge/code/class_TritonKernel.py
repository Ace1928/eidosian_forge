from __future__ import annotations
import collections
import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import operator
import os
import textwrap
from typing import Any, Counter, Dict, Iterable, List, Optional, Set, Tuple, Union
import sympy
import torch
import torch._logging
from torch._prims_common import is_integer_dtype
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch.utils._sympy.value_ranges import ValueRanges
from ..._dynamo.utils import counters
from .. import config, ir, scheduler
from ..codecache import code_hash, get_path, PyCodeCache
from ..dependencies import MemoryDep, StarDep
from ..ir import IRNode, ReductionHint, TritonTemplateBuffer
from ..optimize_indexing import indexing_dtype_strength_reduction
from ..scheduler import BaseScheduling, WhyNoFuse
from ..triton_heuristics import AutotuneHint
from ..utils import (
from ..virtualized import ops, V
from ..wrapper_benchmark import get_kernel_category_by_source_code
from .common import (
from .triton_utils import config_of, signature_of, signature_to_meta
class TritonKernel(Kernel):
    overrides = TritonOverrides
    sexpr = pexpr

    def __init__(self, *groups, index_dtype: str, mutations: Optional[Set[str]]=None, pid_cache=None, reduction_hint=ReductionHint.DEFAULT, min_elem_per_thread=0):
        if pid_cache is None:
            pid_cache = {}
        super().__init__()
        self.numels = [V.graph.sizevars.simplify(s) for s in groups]
        self.mutations: Set[str] = mutations if mutations is not None else set()
        self.range_trees: List[IterationRangesRoot] = []
        self.range_tree_nodes: Dict[sympy.Symbol, IterationRangesEntry] = {}
        self.iter_vars_count = itertools.count()
        self.inside_reduction = self.numels[-1] != 1
        self.body = IndentedBuffer()
        self.indexing_code = IndentedBuffer()
        self.suffix: IndentedBuffer = IndentedBuffer()
        self.outside_loop_vars: Set[Any] = set()
        self.reduction_hint = reduction_hint
        self.index_dtype: str = index_dtype
        self.min_elem_per_thread = min_elem_per_thread
        self.last_usage: Set[str] = set()
        self.persistent_reduction: bool = self.should_use_persistent_reduction()
        self.no_x_dim = self.reduction_hint == ReductionHint.INNER and self.persistent_reduction and (len(self.numels) == 2) and (self.numels[-1] >= 256)
        self.initialize_range_tree(pid_cache)
        self.autotune_hints: Set[AutotuneHint] = set()

        @functools.lru_cache(None)
        def simplify_indexing(index: sympy.Expr):
            index = V.graph.sizevars.simplify_with_ranges(index, self.var_ranges())
            for tree in self.range_trees:
                index = self.combine_contiguous_dims(index, tree)
            return index
        self.simplify_indexing = simplify_indexing

    def need_numel_args(self):
        """
        Indicate whether we need provide numel as arguments for the generated
        kernel calls in the benchmark.

        Should be true for pointwise/reduction kernels but false for triton
        matmul kernels.
        """
        return True

    def should_use_persistent_reduction(self) -> bool:
        """
        Heuristic to set self.persistent_reduction and add guards
        if needed.
        """
        if not (self.inside_reduction and config.triton.persistent_reductions):
            return False
        threshold = {ReductionHint.INNER: 1024}.get(self.reduction_hint, 64)
        last_numel = self.numels[-1]
        if not isinstance(last_numel, (int, sympy.Integer)):
            return False
        hint = V.graph.sizevars.size_hint(last_numel)
        if hint > threshold:
            return False
        V.graph.sizevars.guard_leq(self.numels[-1], next_power_of_2(hint))
        return True

    def set_last_usage(self, nodes):
        if not self.inside_reduction or self.persistent_reduction:
            return
        self.last_usage = set(itertools.chain.from_iterable((n.last_usage for n in nodes if n is not EnableReduction)))

    def initialize_range_tree(self, pid_cache):
        names = list(reversed(['xindex', 'yindex', 'zindex'][:len(self.numels) - 1])) + ['rindex']
        for i in range(len(self.numels)):
            pid_idx = i if names[i][0] == 'r' else 'xyz'.find(names[i][0])
            self.range_trees.append(IterationRangesRoot(names[i], self.numels[i], names[i][0], pid_idx, self, pid_cache))
        for tree in self.range_trees:
            if not tree.is_loop():
                tree.codegen_header(self.body, self.no_x_dim)
        if self.inside_reduction and self.range_trees[-1].is_loop():
            self.body.writeline(f'rbase = {self.range_trees[-1].ranges_code()}')

    def disable_reduction(self):

        @contextlib.contextmanager
        def ctx():
            if self.numels[-1] == 1:
                assert not self.inside_reduction
                yield
                return
            if not self.persistent_reduction:
                self.codegen_body()
            self.inside_reduction = False
            try:
                yield
                if not self.persistent_reduction:
                    self.codegen_body()
            finally:
                self.inside_reduction = True
        return ctx()

    def set_ranges(self, *lengths):
        assert len(lengths) == len(self.range_trees)
        return [ranges.construct(length) for length, ranges in zip(lengths, self.range_trees)]

    @staticmethod
    def _split_iteration_ranges(groups: Iterable[sympy.Expr], lengths: List[List[sympy.Expr]]):
        sv = V.graph.sizevars
        new_ranges: List[List[sympy.Expr]] = [[] for _ in groups]
        remaining = [sv.simplify(g) for g in groups]
        var_count = itertools.count()

        def add_range(i, expr):
            expr = sv.simplify(expr)
            if not sv.statically_known_multiple_of(remaining[i], expr):
                raise CantSplit()
            remaining[i] = FloorDiv(remaining[i], expr)
            new_ranges[i].append(expr)
            return next(var_count)

        def make_combined(size, idx1, idx2):

            def getter(flat_vars):
                return size * flat_vars[idx1] + flat_vars[idx2]
            return getter
        return_getters_groups = []
        current_group = 0
        for length_group in lengths:
            return_getters = []
            for size in length_group:
                if sv.statically_known_equals(size, 1):
                    return_getters.append(lambda _: sympy.Integer(0))
                    continue
                while current_group < len(remaining) and sv.size_hint(remaining[current_group]) == 1:
                    current_group += 1
                if sv.size_hint(size) > sv.size_hint(remaining[current_group]):
                    if not sv.statically_known_multiple_of(size, remaining[current_group]):
                        raise CantSplit()
                    size1 = remaining[current_group]
                    size2 = FloorDiv(size, remaining[current_group])
                    return_getters.append(make_combined(size2, add_range(current_group, size1), add_range(current_group + 1, size2)))
                else:
                    return_getters.append(operator.itemgetter(add_range(current_group, size)))
            return_getters_groups.append(return_getters)
        assert all((V.graph.sizevars.size_hint(s) == 1 for s in remaining)), f'failed to set ranges {remaining} {lengths}'
        return (new_ranges, return_getters_groups)

    @classmethod
    def is_compatible(cls, groups: Iterable[sympy.Expr], lengths: List[List[sympy.Expr]]):
        try:
            cls._split_iteration_ranges(groups, lengths)
            return True
        except CantSplit:
            return False

    def split_and_set_ranges(self, lengths: List[List[sympy.Expr]]):
        """
        We may want to fuse `for i0 in s0*s1` into a tiled kernel with groups (s0, s1).

        To do this we need to split up the iteration space of i0 into something like:
            for i1 in s0:
              for i2 in s1:
                i0 = i1*s1 + i2
                ....

        This function matches and resplits lengths to the groups of
        this kernel to enable tiled + non-tiled fusions.
        """
        groups = [rt.numel for rt in self.range_trees]
        if not self.inside_reduction:
            groups[-1] = sympy.Integer(1)
        if len(lengths) == len(self.range_trees) and all((V.graph.sizevars.simplify(sympy_product(x) - g) == 0 for x, g in zip(lengths, groups))):
            return self.set_ranges(*lengths)
        new_ranges, return_getters_groups = self._split_iteration_ranges(groups, lengths)
        itervars = list(itertools.chain(*self.set_ranges(*new_ranges)))
        return [[fn(itervars) for fn in fns] for fns in return_getters_groups]

    def is_indirect_indexing(self, index: sympy.Expr):
        return free_symbol_startswith(index, 'tmp')

    def is_broadcasted(self, index: sympy.Expr):
        if self.is_indirect_indexing(index):
            return False
        index_numels = [1] * len(self.numels)
        for symbol in index.free_symbols:
            if symbol not in self.range_tree_nodes:
                continue
            entry = self.range_tree_nodes[symbol]
            assert isinstance(entry.parent, IterationRangesRoot)
            index_numels[entry.parent.index] *= entry.length
        simplify = V.graph.sizevars.simplify
        return any((simplify(idx_range) != simplify(iter_range) for idx_range, iter_range in zip(index_numels, self.numels)))

    def combine_contiguous_dims(self, index: sympy.Expr, tree: IterationRangesRoot):
        """
        More aggressive simplification to merge contiguous dims
        """
        if isinstance(index, (sympy.Integer, sympy.Symbol)):
            return index
        index_vars, sizes = tree.vars_and_sizes(index)
        if len(sizes) <= 1:
            return index
        new_sizes, reindex, prune = V.graph.sizevars._simplify_loops(index_vars, sizes, index_prevent_reordering([index], index_vars, sizes))
        if new_sizes == sizes:
            return index
        new_index_vars = tree.construct(new_sizes)
        new_index = sympy_subs(index, dict(zip(index_vars, reindex(new_index_vars))))
        return new_index

    def index_to_str(self, index: sympy.Expr) -> str:
        """
        Convert an index expr to a string that can be used in triton code.
        e.g. a sympy expression "s2" may actually appear as "ks1" in the triton kernel.

        Index expressions often need to be passed in as arguments to the triton kernel.
        Rename_indexing and codegen_indexing keep track of the needed indices and add
        new parameters to the function signature.
        """
        return texpr(self.rename_indexing(self.codegen_indexing(index)))

    def indexing(self, index: sympy.Expr, *, copy_shape=None, dense_indexing=False, override_mask=None):
        """
        Compute the index and mask to pass to tl.load() or tl.store()
        """
        index = self.simplify_indexing(index)
        index = sympy_subs(index, V.graph.sizevars.precomputed_replacements)
        if len(index.atoms(sympy.floor)) or len(index.atoms(sympy.ceiling)):
            index = index.subs(V.graph.sizevars.precomputed_replacements)
        if len(index.atoms(sympy.ceiling)):
            for a in index.atoms(sympy.ceiling):
                symbols = a.free_symbols
                if len(symbols) > 0 and all((s.name.startswith('s') or s.name.startswith('ps') for s in symbols)):
                    replacements = {a: V.graph.sizevars.lookup_precomputed_size(a)}
                    index = sympy_subs(index, replacements)
        index_vars = index.free_symbols
        index = self.simplify_indexing(index)
        index_str = self.index_to_str(index)
        mask_vars: Set[str] = set()
        for var in index_vars:
            assert isinstance(var, sympy.Symbol)
            if override_mask:
                pass
            elif var.name.startswith('tmp'):
                cse_var = self.cse.varname_map[var.name]
                mask_vars.update(cse_var.mask_vars)
            elif var.name.startswith(('s', 'ps', 'i')):
                pass
            else:
                assert var.name[0] in 'xyr', var.name
                mask_vars.add(f'{var.name[0]}mask')
        need_dense = (config.triton.dense_indexing or dense_indexing or self._load_mask is not None) and index != 0
        have_dense = True
        have_loop_vars = False
        dense_mask_vars = set()
        for tree in self.range_trees:
            if tree.prefix == 'r' and (not self.inside_reduction):
                continue
            if index_vars.intersection(tree.var_list):
                have_loop_vars = True
            else:
                have_dense = False
            dense_mask_vars.add(f'{tree.prefix}mask')
        expand_str = None
        if isinstance(index, sympy.Integer):
            expand_str = f'{copy_shape}.shape' if copy_shape else self.dense_size_str()
            index_str = f'tl.full({expand_str}, {index_str}, tl.int32)'
            return (index_str, set(), 'None', expand_str)
        if need_dense and (not have_dense):
            expand_str = f'{copy_shape}.shape' if copy_shape else self.dense_size_str()
            index_str = f'tl.broadcast_to({index_str}, {expand_str})'
            mask_vars = dense_mask_vars
        elif not have_loop_vars and copy_shape:
            index_str = f'tl.broadcast_to({index_str}, {copy_shape}.shape)'
            mask_vars = dense_mask_vars
        if override_mask:
            mask_vars = {override_mask}
        if self._load_mask:
            mask_vars.add(self._load_mask)
        self.filter_masks(mask_vars)
        mask_str = ' & '.join(sorted(map(str, mask_vars))) if mask_vars else 'None'
        return (index_str, mask_vars, mask_str, expand_str)

    def filter_masks(self, mask_vars):
        for tree in self.range_trees:
            if V.graph.sizevars.statically_known_equals(tree.numel, 1):
                mask_vars.discard(f'{tree.prefix}mask')
                continue
            if tree.prefix.upper() not in config.triton.max_block:
                continue
            max_block = config.triton.max_block[tree.prefix.upper()]
            if V.graph.sizevars.statically_known_multiple_of(tree.numel, max_block):
                mask_vars.discard(f'{tree.prefix}mask')

    def var_ranges(self):
        return dict(itertools.chain.from_iterable((tree.var_ranges.items() for tree in self.range_trees)))

    def codegen_indexing(self, expr: sympy.Expr):
        expr = V.graph.sizevars.simplify_with_ranges(expr, self.var_ranges())
        for sym in sorted(expr.free_symbols, key=str):
            if sym in self.range_tree_nodes:
                replacements = {}
                for ps in self.range_tree_nodes[sym].precomputed_args():
                    replacements[ps] = V.graph.sizevars.lookup_precomputed_size(ps)
                if len(replacements) > 0:
                    self.range_tree_nodes[sym].expr = sympy_subs(self.range_tree_nodes[sym].expr, replacements)
                self.range_tree_nodes[sym].codegen()
        return expr

    @contextlib.contextmanager
    def mask_loads(self, mask):
        """Context manager to add an additional mask to tl.load/store"""
        prior = self._load_mask
        if prior:
            mask = self.cse.generate(self.compute, f'{mask} & {prior}')
        self._load_mask = mask
        try:
            yield mask
        finally:
            self._load_mask = prior

    def generate_assert(self, check):
        return torch.version.hip is None and super().generate_assert(check)

    def load_mask(self, var):
        mask = ''
        mask_vars = set(var.mask_vars)
        if self._load_mask:
            mask_vars.add(self._load_mask)
        if mask_vars:
            mask = f'{next(iter(mask_vars))}' if len(mask_vars) == 1 else f'({' & '.join((str(v) for v in mask_vars))})'
        return mask

    @property
    def assert_function(self) -> str:
        return 'tl.device_assert'

    def get_strides_of_load(self, index: sympy.Expr):
        """
        This gets the stride of the index for each of the tiling variables
        (technically, it does it at index 0)

        For example, if
        xindex = x0 + 512*x1 + 1024*r0
        x0 = (xindex//512)
        x1 = (xindex % 512)
        r0 = rindex // 1024

        this function would return
        {xindex: 512, rindex: 1024}
        """
        index_to_tile_indexes = {k: v.expr for k, v in self.range_tree_nodes.items()}
        index_in_tile_vars = sympy_subs(index, index_to_tile_indexes)
        strides = {}
        for range_tree in self.range_trees:
            s = sympy_symbol(range_tree.name)
            strides[s] = sympy_subs(index_in_tile_vars, {s: 1}) - sympy_subs(index_in_tile_vars, {s: 0})
        return strides

    def load(self, name: str, index: sympy.Expr):
        var = self.args.input(name)
        indirect_indexing = self.is_indirect_indexing(index)
        original_index = index
        index, mask_vars, mask, expand_str = self.indexing(index)
        is_coalesced = any((i == 1 for i in self.get_strides_of_load(original_index).values()))
        if self.is_broadcasted(original_index):
            ep = ", eviction_policy='evict_last'"
        elif not is_coalesced:
            ep = ", eviction_policy='evict_last'"
        elif self.inside_reduction and (not self.persistent_reduction):
            if name in self.args.inplace_buffers:
                names = set(self.args.inplace_buffers[name].other_names)
            else:
                names = {name}
            last_use = len(names & self.last_usage) > 0
            evict_last = not last_use and ('rmask' in mask or indirect_indexing)
            if evict_last:
                ep = ", eviction_policy='evict_last'"
            else:
                ep = ", eviction_policy='evict_first'"
        else:
            ep = ''
        if ('tmp' in mask or 'rmask' in mask) and V.graph.get_dtype(name) != torch.bool:
            other = ', other=0.0'
        else:
            other = ''
        append_broadcast = None
        if V.graph.is_unspec_arg(name):
            line = var
        else:
            if isinstance(original_index, sympy.Integer):
                line = f'tl.load({var} + ({original_index}))'
                append_broadcast = expand_str
            else:
                line = f'tl.load({var} + ({index}), {mask}{ep}{other})'
            dtype = V.graph.get_dtype(name)
            if dtype in (torch.float16, torch.bfloat16):
                line += '.to(tl.float32)'
            if dtype == torch.bool and torch.version.hip is None:
                line += '.to(tl.int1)'
        if 'tmp' in mask:
            load_buffer = self.compute
        elif self.inside_reduction and (not self.persistent_reduction) and ('rmask' not in mask) and (not indirect_indexing):
            load_buffer = self.body
        else:
            load_buffer = self.loads
        result_var = self.cse.generate(load_buffer, line)
        assert isinstance(result_var, TritonCSEVariable)
        result_var.mask_vars = mask_vars
        if append_broadcast:
            line = f'tl.broadcast_to({result_var}, {append_broadcast})'
            result_var = self.cse.generate(load_buffer, line)
        if not self.inside_reduction or 'rmask' not in mask:
            self.outside_loop_vars.add(result_var)
        return result_var

    def store(self, name, index, value, mode=None):
        var = self.args.output(name)
        indirect_indexing = self.is_indirect_indexing(index)
        original_index = index
        index, mask_vars, mask, expand_str = self.indexing(index, dense_indexing=True)
        is_inplace = name in self.args.inplace_buffers
        is_broadcasted = self.is_broadcasted(original_index)
        if is_inplace and is_broadcasted:
            self.stores.writeline(DeferredLine(name, 'tl.debug_barrier()'))
        if mode is None:
            line = f'tl.store({var} + ({index}), {value}, {mask})'
        elif mode == 'atomic_add':
            line = f'tl.atomic_add({var} + ({index}), {value}, {mask})'
        else:
            raise NotImplementedError(f'store mode={mode}')
        self.stores.writeline(DeferredLine(name, line))
        if not self.inside_reduction:
            self.outside_loop_vars.add(value)

    def bucketize(self, values: CSEVariable, offsets_name: str, offsets_size: sympy.Expr, indexing_dtype: torch.dtype, right: bool):
        """
        See [Note: Inductor bucketize op]
        """
        self.autotune_hints.add(AutotuneHint.ELEMENTS_PER_WARP_32)
        offsets_ptr = self.args.input(offsets_name)
        block_size = self.dense_size_str()
        offsets_size_str = self.index_to_str(offsets_size)
        if indexing_dtype == torch.int32:
            triton_dtype = 'tl.int32'
        elif indexing_dtype == torch.int64:
            triton_dtype = 'tl.int64'
        else:
            raise NotImplementedError('Bucketize only supports indexing with int32 and int64')
        result = self.cse.generate(self.compute, f'triton_helpers.bucketize_binary_search({values}, {offsets_ptr}, {triton_dtype}, {right}, {offsets_size_str}, {block_size})')
        return result

    def reduction_resize(self, value):
        ndims = self.triton_tensor_ndim()
        if ndims == 1:
            return f'triton_helpers.promote_to_tensor({value})'
        sizes = [':'] * ndims
        sizes[-1] = 'None'
        return f'{value}[{', '.join(sizes)}]'

    @staticmethod
    def _map_tuple_or_scalar(fn, value):
        if isinstance(value, tuple):
            return tuple(map(fn, value))
        return fn(value)

    def reduction(self, dtype, src_dtype, reduction_type, value):
        assert self.inside_reduction
        masks = {f'{tree.prefix}mask' for tree in self.range_trees}
        self.filter_masks(masks)
        masks = sorted(masks)
        if self._load_mask:
            masks.append(self._load_mask)
        reduction_range_prefix = self.range_trees[-1].prefix
        reduction_sizes = ['None' for _ in self.range_trees]
        reduction_sizes[-1] = ':'
        dense_size_str = self.dense_size_str()
        value = self._map_tuple_or_scalar(lambda v: self.cse.generate(self.compute, f'tl.broadcast_to({v}, {dense_size_str})'), value)
        dim: int
        root_op: str

        def final_reduction(value):
            use_helper = reduction_type in {'any', 'max', 'min', 'prod'}
            module = 'triton_helpers' if use_helper else 'tl'
            if reduction_type in {'max', 'min'}:
                return self.reduction_resize(f'{module}.{reduction_type}2({value}, {dim})')
            return self.reduction_resize(f'{module}.{reduction_type}({value}, {dim})')

        def final_argreduce(buffer, result_var, value, index):
            buffer.splice(f'                _, {result_var}_tmp = triton_helpers.{root_op}_with_index({value}, {index}, {dim})\n                {result_var} = {self.reduction_resize(f'{result_var}_tmp')}\n                ')
        cache_key = (src_dtype, reduction_type, value)
        if cache_key in self.cse.reduction_cache:
            return self.cse.reduction_cache[cache_key]
        dim = len(self.range_trees) - 1 - int(bool(self.no_x_dim))
        acc_type = triton_acc_type(src_dtype)
        result_var: Any = self.cse.newvar()
        result_var.mask_vars = {var for var in masks if var[0] != 'r'}
        cond = ' & '.join(masks)
        if self.persistent_reduction:
            default = ir.Reduction.default_value(reduction_type, src_dtype)
            default = self._map_tuple_or_scalar(triton_constant, default)

            def _mask_value(value, default):
                return self.cse.generate(self.compute, f'tl.where({cond}, {value}, {default})')
            if isinstance(value, tuple):
                masked_value = [_mask_value(v, d) for v, d in zip(value, default)]
            else:
                masked_value = _mask_value(value, default)
            if reduction_type in {'argmax', 'argmin'}:
                accumulator_index = str(self.cse.generate(self.compute, f'tl.broadcast_to({reduction_range_prefix}index, {masked_value}.shape)'))
                root_op = {'argmax': 'max', 'argmin': 'min'}[reduction_type]
                final_argreduce(self.compute, result_var, masked_value, accumulator_index)
            elif reduction_type == 'welford_reduce':
                sum_ = ops.reduction(dtype, dtype, 'sum', value)
                self.inside_reduction = False
                rnumel = ops.index_expr(self.numels[-1], dtype)
                mean = ops.truediv(sum_, rnumel)
                self.inside_reduction = True
                dx = ops.sub(value, mean)
                dx2 = ops.mul(dx, dx)
                m2 = ops.reduction(dtype, dtype, 'sum', dx2)
                result_var = (mean, m2, rnumel)
            elif reduction_type == 'welford_combine':
                mean, m2, weight = masked_value
                welford = f'triton_helpers.welford({mean}, {m2}, {weight}, {dim})'
                mean, m2, weight = (self.cse.newvar() for _ in range(3))
                self.compute.writeline(f'{mean}, {m2}, {weight} = {welford}')
                result_var = tuple((self.cse.generate(self.compute, self.reduction_resize(var_name)) for var_name in (mean, m2, weight)))
            else:
                result_var = self.cse.generate(self.compute, final_reduction(masked_value))
        else:
            accumulator = f'_{result_var}'
            default = ir.Reduction.default_accumulator(reduction_type, src_dtype)
            default = self._map_tuple_or_scalar(triton_constant, default)
            if not isinstance(default, tuple):
                self.body.writeline(f'{accumulator} = tl.full({self.dense_size_str()}, {default}, {acc_type})')
            if reduction_type in {'argmax', 'argmin'}:
                accumulator_index = f'_{result_var}_index'
                long_max = torch.iinfo(torch.int64).max
                self.body.writeline(f'{accumulator_index} = tl.full({self.dense_size_str()}, {long_max}, tl.int64)')
                root_op = {'argmax': 'max', 'argmin': 'min'}[reduction_type]
                self.compute.splice(f'                {accumulator}_next, {accumulator_index}_next = triton_helpers.{root_op}imum_with_index(\n                    {accumulator}, {accumulator_index}, {value}, {reduction_range_prefix}index\n                )\n                {accumulator} = tl.where({cond}, {accumulator}_next, {accumulator})\n                {accumulator_index} = tl.where({cond}, {accumulator_index}_next, {accumulator_index})\n                ')
                final_argreduce(self.suffix, result_var, accumulator, accumulator_index)
            elif is_welford_reduction(reduction_type):
                accumulator = f'{result_var}_mean'
                accumulator_m2 = f'{result_var}_m2'
                accumulator_weight = f'{result_var}_weight'
                self.body.writeline(f'{accumulator} = tl.zeros({self.dense_size_str()}, {acc_type})')
                self.body.writeline(f'{accumulator_m2} = tl.zeros({self.dense_size_str()}, {acc_type})')
                self.body.writeline(f'{accumulator_weight} = tl.zeros({self.dense_size_str()}, {acc_type})')
                if reduction_type == 'welford_combine':
                    mean, m2, weight = value
                    self.compute.splice(f'                    {accumulator}_next, {accumulator_m2}_next, {accumulator_weight}_next = triton_helpers.welford_combine(\n                        {accumulator}, {accumulator_m2}, {accumulator_weight},\n                        {mean}, {m2}, {weight}\n                    )\n                    ')
                else:
                    assert reduction_type == 'welford_reduce'
                    self.compute.splice(f'                    {accumulator}_next, {accumulator_m2}_next, {accumulator_weight}_next = triton_helpers.welford_reduce(\n                        {value}, {accumulator}, {accumulator_m2}, {accumulator_weight},\n                    )\n                    ')
                self.compute.splice(f'                {accumulator} = tl.where({cond}, {accumulator}_next, {accumulator})\n                {accumulator_m2} = tl.where({cond}, {accumulator_m2}_next, {accumulator_m2})\n                {accumulator_weight} = tl.where({cond}, {accumulator_weight}_next, {accumulator_weight})\n                ')
                result_mean = result_var
                result_m2 = self.cse.newvar()
                result_weight = self.cse.newvar()
                self.suffix.splice(f'                {result_mean}_tmp, {result_m2}_tmp, {result_weight}_tmp = triton_helpers.welford(\n                    {accumulator}, {accumulator_m2}, {accumulator_weight}, {dim}\n                )\n                {result_mean} = {self.reduction_resize(f'{result_mean}_tmp')}\n                {result_m2} = {self.reduction_resize(f'{result_m2}_tmp')}\n                {result_weight} = {self.reduction_resize(f'{result_weight}_tmp')}\n                ')
                result_var = (result_mean, result_m2, result_weight)
            else:
                combine_fn = ir.get_reduction_combine_fn(reduction_type, src_dtype)
                updated = combine_fn(accumulator, value)
                self.compute.writeline(f'{accumulator} = tl.where({cond}, {updated}, {accumulator})')
                if src_dtype == torch.bool:
                    accumulator = f'{accumulator}.to(tl.int8)'
                    result_type = triton_compute_type(dtype)
                    self.suffix.writeline(f'{result_var} = {final_reduction(accumulator)}.to({result_type})')
                else:
                    self.suffix.writeline(f'{result_var} = {final_reduction(accumulator)}')
        self.cse.reduction_cache[cache_key] = result_var
        if isinstance(result_var, tuple):
            self.outside_loop_vars |= set(result_var)
        else:
            self.outside_loop_vars.add(result_var)
        return result_var

    def store_reduction(self, name, index, value):
        assert self.inside_reduction
        self.inside_reduction = False
        index, mask_vars, mask, _ = self.indexing(index)
        assert 'rmask' not in index
        self.inside_reduction = True
        var = self.args.output(name)
        self.suffix.writeline(DeferredLine(name, f'tl.store({var} + ({index}), {value}, {mask})'))

    def codegen_body(self):
        """
        Concat output code from index_code, loads, compute, stores,
        suffix into self.body.

        For pointwise kernels, this is called just once at the end.

        For reduction kernels, this generates a loop over the reduction
        axis.
        """
        if not (self.indexing_code or self.loads or self.stores or self.compute or self.suffix):
            return
        if self.inside_reduction and (not self.persistent_reduction):
            self.body.writeline('for roffset in range(0, rnumel, RBLOCK):')
            with self.body.indent():
                self.range_trees[-1].codegen_header(self.body)
                self.body.splice(self.indexing_code)
                self.body.splice(self.loads)
                self.body.splice(self.compute)
                self.body.splice(self.stores)
            self.cse.invalidate(self.outside_loop_vars)
            self.range_trees[-1].cache_clear()
        else:
            self.body.splice(self.indexing_code)
            self.body.splice(self.loads)
            self.body.splice(self.compute)
            self.body.splice(self.stores)
        self.body.splice(self.suffix)
        self.indexing_code.clear()
        self.loads.clear()
        self.compute.clear()
        self.stores.clear()
        self.suffix.clear()

    def codegen_kernel_benchmark(self):
        result = IndentedBuffer()
        argdefs, call_args, signature = self.args.python_argdefs()
        result.writelines(['', '', 'def get_args():'])
        with result.indent():
            name_cnt = itertools.count()
            var_names = []
            for arg_name, arg_sig in zip(call_args, signature):
                var_name = f'arg_{next(name_cnt)}'
                buf = V.graph.get_buffer(arg_name)
                if buf:
                    result.writeline(f"{var_name} = rand_strided({V.graph.sizevars.size_hints(buf.get_size())}, {V.graph.sizevars.size_hints(buf.get_stride())}, device='{buf.get_device()}', dtype={buf.get_dtype()})")
                elif arg_name in V.graph.constants:
                    const_tensor = V.graph.constants[arg_name]
                    result.writeline(f"{var_name} = rand_strided({V.graph.sizevars.size_hints(const_tensor.size())}, {V.graph.sizevars.size_hints(const_tensor.stride())}, device='{const_tensor.device}', dtype={const_tensor.dtype})")
                elif isinstance(arg_sig, SizeArg):
                    symval_hint = V.graph.sizevars.size_hint(arg_sig.expr)
                    if 'seed_offset' in arg_sig.name:
                        symval_hint = 0
                    result.writeline(f'{var_name} = {symval_hint}')
                else:
                    raise KeyError(f"Don't find the buffer or const tensor for {arg_name}")
                var_names.append(var_name)
            result.writeline(f'return {', '.join(var_names)},')
        result.writelines(['\n', '\n', 'def call(args):'])
        grid = []
        extra_args = []
        extra_args_str = None
        index = V.graph.scheduler.current_device.index
        with result.indent():
            result.writeline(f'with torch.cuda._DeviceGuard({index}):')
            with result.indent():
                result.writeline(f'torch.cuda.set_device({index})')
                for tree in self.range_trees:
                    expr = pexpr(V.graph.sizevars.size_hint(tree.numel))
                    if tree.prefix != 'r' or self.inside_reduction:
                        extra_args.append(expr)
                    if tree.prefix != 'r':
                        grid.append(expr)
                stream_name = f'stream{index}'
                result.writeline(f'{stream_name} = get_cuda_stream({index})')
                if self.need_numel_args():
                    extra_args_str = ', '.join(map(str, extra_args)) + ', '
                else:
                    extra_args_str = ''
                result.writeline(f'{str(Placeholder.KERNEL_NAME)}.run(*args, {extra_args_str}grid=grid({', '.join(grid)}), stream={stream_name})')
        result.writelines(['\n', '\n', 'def benchmark_all_configs(args):'])
        with result.indent():
            result.writeline(f'with torch.cuda._DeviceGuard({index}):')
            with result.indent():
                result.writeline(f'torch.cuda.set_device({index})')
                result.writeline(f'return {str(Placeholder.KERNEL_NAME)}.benchmark_all_configs(*args, {extra_args_str}grid=grid({', '.join(grid)}))')
        ninplace_args = len(unique(self.args.inplace_buffers.values()))
        result.writelines(['\n', '\n', "if __name__ == '__main__':"])
        with result.indent():
            result.writeline('from torch._inductor.utils import get_num_bytes')
            result.writeline('from triton.testing import do_bench')
            result.writeline('')
            result.writeline('args = get_args()')
            result.writeline('ms = do_bench(lambda: call(args), rep=40, fast_flush=True)')
            result.writeline(f'num_gb = get_num_bytes(*args, num_in_out_args={ninplace_args}) / 1e9')
            result.writeline('gb_per_s = num_gb / (ms / 1e3)')
            result.writeline('print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")')
        return result

    def imports_for_benchmark_kernel(self):
        return textwrap.dedent('\n            from torch._dynamo.testing import rand_strided\n            from torch._C import _cuda_getCurrentRawStream as get_cuda_stream\n            import torch\n            from torch._inductor.triton_heuristics import grid\n        ')

    def codegen_kernel(self, name=None):
        from triton import next_power_of_2
        code = IndentedBuffer()
        size_hints = []
        for numel in self.numels:
            numel_hint = V.graph.sizevars.symbolic_hint(numel)
            if not isinstance(numel_hint, (int, sympy.Integer)):
                size_hint = 8192
            else:
                size_hint = next_power_of_2(int(numel_hint))
            size_hints.append(size_hint)
        if self.persistent_reduction:
            assert self.inside_reduction
            heuristics = 'persistent_reduction'
        elif self.inside_reduction:
            heuristics = 'reduction'
        else:
            size_hints.pop()
            heuristics = 'pointwise'
        if name is None:
            code.splice(f'\n                    import triton\n                    import triton.language as tl\n                    from torch._inductor.ir import ReductionHint\n                    from torch._inductor.ir import TileHint\n                    from torch._inductor.triton_heuristics import AutotuneHint, {heuristics}\n                    from torch._inductor.utils import instance_descriptor\n                    from torch._inductor import triton_helpers\n                ')
            if config.benchmark_kernel:
                code.splice(self.imports_for_benchmark_kernel())
        argdefs, _, signature = self.args.python_argdefs()
        for i, arg in enumerate(signature):
            if isinstance(arg, SizeArg) and arg.expr in V.graph.sizevars.inv_precomputed_replacements:
                signature[i] = SizeArg(arg.name, V.graph.sizevars.inv_precomputed_replacements[arg.expr])
        mutated_args = set()
        for mutation in self.mutations:
            if mutation in self.args.input_buffers:
                mutated_args.add(self.args.input_buffers[mutation])
            if mutation in self.args.inplace_buffers and mutation not in V.graph.removed_buffers and (mutation not in self.removed_buffers):
                mutated_args.add(self.args.inplace_buffers[mutation].inner_name)
            if mutation in self.args.output_buffers:
                mutated_args.add(self.args.output_buffers[mutation])
        mutated_args = sorted(mutated_args)
        triton_meta_signature = signature_to_meta(signature, size_dtype=self.index_dtype)
        triton_meta = {'signature': triton_meta_signature, 'device': V.graph.scheduler.current_device.index, 'device_type': V.graph.scheduler.current_device.type, 'constants': {}}
        inductor_meta = {'autotune_hints': set(self.autotune_hints), 'kernel_name': str(Placeholder.DESCRIPTIVE_NAME), 'mutated_arg_names': mutated_args}
        for tree in self.range_trees:
            if tree.prefix != 'r' or self.inside_reduction:
                sizearg = SizeArg(f'{tree.prefix}numel', tree.numel)
                signature.append(sizearg)
                triton_meta_signature[len(argdefs)] = signature_of(sizearg, size_dtype=self.index_dtype)
                argdefs.append(f'{tree.prefix}numel')
        triton_meta['configs'] = [config_of(signature)]
        for tree in self.range_trees:
            if tree.prefix == 'r' and (not self.inside_reduction or self.persistent_reduction):
                continue
            if tree.prefix == 'x' and self.no_x_dim:
                continue
            argdefs.append(f'{tree.prefix.upper()}BLOCK : tl.constexpr')
        if self.inside_reduction:
            reduction_hint = self.reduction_hint
            heuristics_line = f'\n                @{heuristics}(\n                    size_hints={size_hints!r},\n                    reduction_hint={reduction_hint},\n                    filename=__file__,\n                    triton_meta={triton_meta!r},\n                    inductor_meta={inductor_meta!r}\n                )\n                @triton.jit\n            '
        else:
            tile_hint = ''
            if len(size_hints) == 2:
                if len(signature) == 4:
                    tile_hint = 'tile_hint=TileHint.SQUARE,'
                else:
                    tile_hint = 'tile_hint=TileHint.DEFAULT,'
            heuristics_line = f'\n                @{heuristics}(\n                    size_hints={size_hints!r}, {tile_hint}\n                    filename=__file__,\n                    triton_meta={triton_meta!r},\n                    inductor_meta={inductor_meta!r},\n                    min_elem_per_thread={self.min_elem_per_thread}\n                )\n                @triton.jit\n            '
        code.splice(heuristics_line)
        code.writeline(f'def {name or str(Placeholder.KERNEL_NAME)}({', '.join(argdefs)}):')
        self.codegen_body()
        with code.indent():
            self.codegen_static_numels(code)
            for old, new in self.args.aliases():
                code.writeline(f'{old} = {new}')
            code.splice(self.body)
        if config.benchmark_kernel:
            code.splice(self.codegen_kernel_benchmark())
        return code.getvalue()

    def codegen_static_numels(self, code):
        """
        We get a small speedup from hard coding numels if they are static.

        This code stomps on the passed-in values by writing an constant to the top of the kernel.

        In a kernel like:
        def KERNEL_NAME(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):

        We would add
        xnumel = 4096
        rnumel = 768

        After the signature, before the kernel code, if we decided to make these static. As its hardcoded, it becomes
        a better signal to triton on how to unroll and do some static indexing. So, it's not so much that downstream
        knows that its a static numel, as that you just plop a constant into the kernel.
        """
        for tree in self.range_trees:
            if tree.prefix != 'r' or self.inside_reduction:
                simplified_tree_numel = V.graph.sizevars.simplify(tree.numel)
                if isinstance(simplified_tree_numel, (sympy.Integer, int)):
                    code.writeline(f'{tree.prefix}numel = {int(simplified_tree_numel)}')
            if tree.prefix == 'r' and self.persistent_reduction:
                simplified_tree_numel = V.graph.sizevars.simplify(tree.numel)
                if isinstance(simplified_tree_numel, (sympy.Integer, int)):
                    val = int(simplified_tree_numel)
                else:
                    continue
                val = next_power_of_2(val)
                code.writeline(f'RBLOCK: tl.constexpr = {val}')
            if tree.prefix == 'x' and self.no_x_dim:
                code.writeline('XBLOCK: tl.constexpr = 1')

    def triton_tensor_ndim(self):
        no_x_dim = int(bool(self.no_x_dim))
        no_r_dim = self.numels[-1] == 1
        return len(self.range_trees) - no_x_dim - no_r_dim

    def indexing_size_str(self, i=None, x=None):
        no_x_dim = int(bool(self.no_x_dim))
        sizes = ['None'] * self.triton_tensor_ndim()
        if i is not None:
            idx = i - no_x_dim
            sizes[idx] = ':'
        return f'[{', '.join(sizes)}]'

    def dense_size_str(self):
        sizes = []
        for tree in self.range_trees:
            if self.no_x_dim and tree.prefix == 'x':
                continue
            if tree.prefix != 'r' or self.inside_reduction:
                sizes.append(f'{tree.prefix.upper()}BLOCK')
            elif tree.prefix == 'r' and tree.numel != 1:
                sizes.append('1')
        if sizes[0:3] == ['ZBLOCK', 'YBLOCK', 'XBLOCK']:
            sizes[0:3] = reversed(sizes[0:3])
        if sizes[0:2] == ['YBLOCK', 'XBLOCK']:
            sizes[0:2] = reversed(sizes[0:2])
        return f'[{', '.join(sizes)}]'

    def call_kernel(self, name: str, node: Optional[IRNode]=None):
        wrapper = V.graph.wrapper_code
        _, call_args, _ = self.args.python_argdefs()
        for i in range(len(call_args)):
            if V.graph.is_unspec_arg(call_args[i]):
                call_args[i] = call_args[i] + '.item()'
        grid = []
        for tree in self.range_trees:
            if isinstance(tree.numel, (sympy.Integer, sympy.Symbol)):
                expr = tree.numel
            else:
                expr = wrapper.generate_numel_expr(name, tree)
            if tree.prefix != 'r' or self.inside_reduction:
                call_args.append(expr)
            if tree.prefix != 'r':
                grid.append(expr)
        grid = wrapper.generate_default_grid(name, grid)
        wrapper.generate_kernel_call(name, call_args, grid, V.graph.scheduler.current_device.index, cuda=True, triton=True)

    def codegen_nan_check(self):
        if not config.nan_asserts:
            return
        wrapper = V.graph.wrapper_code
        _, call_args, arg_types = self.args.python_argdefs()
        for arg, arg_type in zip(call_args, arg_types):
            if isinstance(arg_type, TensorArg):
                line = f'assert not {arg}.isnan().any().item()'
                wrapper.writeline(line)
                line = f'assert not {arg}.isinf().any().item()'
                wrapper.writeline(line)

    def warn_mix_layout(self, kernel_name):
        """
        Print message if the kernel have mixed layout inputs.
        Only care about 4D tensor for now.
        """
        if len(self.args.input_buffers) == 1 and len(self.args.output_buffers) == 1 and (len(self.args.inplace_buffers) == 0):
            return
        argdefs, call_args, signature = self.args.python_argdefs()
        uniform_stride_order = None
        for arg_name in call_args:
            buf = V.graph.get_buffer(arg_name)
            if buf and len(buf.layout.size) == 4:
                if len([x for x in buf.layout.size if x == 1]) == 3:
                    continue
                stride_order = ir.get_stride_order(buf.layout.stride)
                if uniform_stride_order is None:
                    uniform_stride_order = stride_order
                elif uniform_stride_order != stride_order:
                    msg = yellow_text(f'Expected stride order {uniform_stride_order}, but found stride order' + f' {stride_order} for kernel {kernel_name}')
                    log.warning(msg)
                    stride_order_list = [ir.get_stride_order(V.graph.get_buffer(name).layout.stride) if V.graph.get_buffer(name) else None for name in call_args]
                    size_list = [V.graph.get_buffer(name).layout.size if V.graph.get_buffer(name) else None for name in call_args]
                    source_list = ['GraphInput' if name in V.graph.graph_inputs else 'IntermediateBuffer' if name in V.graph.name_to_buffer else None for name in call_args]
                    msg = yellow_text(f'  param names {argdefs}\n  buf names {call_args}\n  strides {stride_order_list}' + f'\n  sizes {size_list}\n  sources {source_list}\n')
                    log.warning(msg)
                    return
        msg = green_text(f'All the inputs for the triton kernel {kernel_name} have uniform layout')
        log.warning(msg)

    def create_cse_var(self, *args, **kwargs):
        return TritonCSEVariable(*args, **kwargs)