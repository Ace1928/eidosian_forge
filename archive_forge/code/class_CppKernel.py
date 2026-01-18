import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import re
import sys
from copy import copy, deepcopy
from typing import Dict, List, Optional, Set, Tuple, Union
import sympy
import torch
import torch.fx
from torch._inductor import dependencies
from torch._inductor.ir import StorageBox, TensorBox
from torch._prims_common import is_float_dtype
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges
from .. import codecache, config, ir, metrics
from ..codegen.wrapper import WrapperCodeGen
from ..optimize_indexing import range_expressable_in_32_bits
from ..scheduler import BaseScheduling, SchedulerNode
from ..utils import (
from ..virtualized import ops, V
from .common import (
class CppKernel(Kernel):
    overrides = CppOverrides
    sexpr = cexpr
    newvar_prefix = 'auto '
    suffix = ';'

    def __init__(self, args, num_threads):
        super().__init__(args)
        self.call_ranges: Optional[Tuple[sympy.Expr, ...]] = None
        self.ranges: List[sympy.Expr] = []
        self.itervars: List[sympy.Symbol] = []
        self.reduction_depth = None
        self.reduction_prefix = IndentedBuffer()
        self.reduction_suffix = IndentedBuffer()
        self.reduction_var_map = {}
        self.reduction_cse = CSE(self.newvar_prefix, self.suffix, name_prefix='tmp_acc')
        self.preloads = IndentedBuffer()
        self.poststores = IndentedBuffer()
        self.num_threads = num_threads
        self.reduction_omp_dec: Dict[Tuple[str, str], str] = {}

    @contextlib.contextmanager
    def masked(self, mask):
        """Context manager to add an additional mask to loads and stores."""
        prior = self._load_mask
        if prior:
            mask = self.cse.generate(self.compute, f'{mask} & {prior}')
        self._load_mask = mask
        try:
            yield mask
        finally:
            self._load_mask = prior

    def scale_index_with_offset(self, index: sympy.Expr, scale=1, itervar_idx=-1, offset=0):
        var = self.itervars[itervar_idx]
        replacement = {var: var * scale + offset}
        new_index = sympy_subs(index, replacement)
        return new_index

    def index_to_str(self, index: sympy.Expr) -> str:
        """
        Convert an index expr to a string that can be used in cpp code.
        e.g. a sympy expression "s2" may actually appear as "ks1" in the cpp kernel.
        """
        return cexpr(self.rename_indexing(index))

    def load(self, name: str, index: sympy.Expr):
        var = self.args.input(name)
        index = self.rename_indexing(index)
        line = f'{var}[{cexpr_index(index)}]'
        if V.graph.get_dtype(name) in [torch.float16]:
            line = f'static_cast<float>({line})'
        csevar = self.cse.generate(self.loads, line)
        csevar.update_on_args('load', (name, index), {})
        return csevar

    def store(self, name, index, value, mode=None):
        assert 'buf' in name
        var = self.args.output(name)
        index = self.rename_indexing(index)
        if mode is None:
            line = f'{var}[{cexpr_index(index)}] = {value};'
        elif mode == 'atomic_add':
            if not config.cpp.dynamic_threads and self.num_threads == 1:
                line = f'{var}[{cexpr_index(index)}] += {value};'
            else:
                line = f'atomic_add(&{var}[{cexpr_index(index)}], {value});'
        else:
            raise NotImplementedError(f'store mode={mode}')
        self.stores.writeline(DeferredLine(name, line))

    def reduction(self, dtype, src_dtype, reduction_type, value):
        argmax_or_argmin = reduction_type in {'argmax', 'argmin'}
        reduction_key = (src_dtype, reduction_type, value)
        if reduction_key in self.reduction_cse.reduction_cache:
            return self.reduction_cse.reduction_cache[reduction_key]
        acc = self.reduction_cse.generate(self.loads, f'reduction {reduction_key}', write=False)
        self.reduction_var_map[acc] = reduction_type
        if argmax_or_argmin:
            self.reduction_prefix.writelines(argmax_argmin_prefix(reduction_type, src_dtype, acc))
            compare_op = '<' if reduction_type == 'argmax' else '>'
            assert self.reduction_depth is not None
            index = self.itervars[self.reduction_depth]
            for i in range(self.reduction_depth + 1, len(self.itervars)):
                index = index * self.ranges[i] + self.itervars[i]
            self.stores.writelines([f'if ({acc}.value {compare_op} {value}) {{', f'    {acc}.index = {cexpr_index(index)}; {acc}.value = {value};', '}'])
        else:
            acc_type = reduction_acc_type(reduction_type, dtype)
            if (reduction_type, acc_type) not in self.reduction_omp_dec:
                if RTYPE_TO_CPP[reduction_type] not in NATIVE_OMP_RTYPES:
                    self.reduction_prefix.splice(f'    #pragma omp declare reduction(    {RTYPE_TO_CPP[reduction_type]}:{acc_type}:    omp_out = {reduction_combine(reduction_type, 'omp_out', 'omp_in')})     initializer(omp_priv={{{reduction_init(reduction_type, dtype)}}})\n                ')
                self.reduction_omp_dec[reduction_type, acc_type] = RTYPE_TO_CPP[reduction_type]
            self.reduction_prefix.writeline(f'{acc_type} {acc} = {reduction_init(reduction_type, dtype)};')
            self.stores.writeline(f'{acc} = {reduction_combine(reduction_type, acc, value)};')
        result = reduction_project(reduction_type, acc)
        self.reduction_cse.reduction_cache[reduction_key] = result
        return result

    def store_reduction(self, name, index, value):
        index = self.rename_indexing(index)
        var = self.args.output(name)
        self.reduction_suffix.writeline(DeferredLine(name, f'{var}[{cexpr_index(index)}] = {value};'))

    def set_ranges(self, lengths, reduction_lengths):
        if self.call_ranges:
            assert self.call_ranges == tuple(lengths) + tuple(reduction_lengths), f'{self.call_ranges} == {tuple(lengths)} + {tuple(reduction_lengths)}'
            assert self.reduction_depth == len(lengths)
        else:
            self.call_ranges = tuple(lengths) + tuple(reduction_lengths)
            self.ranges = [self.rename_indexing(x) for x in self.call_ranges]
            self.itervars = [sympy_symbol(f'x{n}') for n in range(len(self.ranges))]
            self.reduction_depth = len(lengths)
        return (self.itervars[:self.reduction_depth], self.itervars[self.reduction_depth:])

    def size_hint(self):
        return V.graph.sizevars.size_hint(sympy_product(self.call_ranges), fallback=8192)

    def codegen_loops_impl(self, loop_nest, code, worksharing):
        threads = parallel_num_threads()
        assert self.call_ranges is not None
        par_depth = self.decide_parallel_depth(self.call_ranges[:loop_nest.max_parallel_depth()], threads)
        with contextlib.ExitStack() as stack:
            if par_depth:
                if loop_nest.is_reduction_only():
                    worksharing.close()
                else:
                    worksharing.parallel(threads)
                loop_nest.mark_parallel(par_depth)
            elif threads > 1:
                if worksharing.single():
                    stack.enter_context(code.indent())

            def gen_kernel(kernel):
                with contextlib.ExitStack() as stack:
                    assert kernel
                    if hasattr(kernel, 'codegen_inner_loops'):
                        code.splice(kernel.preloads)
                        kernel.codegen_inner_loops(code)
                        stack.enter_context(code.indent())
                    code.splice(kernel.loads)
                    code.splice(kernel.compute)
                    code.splice(kernel.stores)
                if hasattr(kernel, 'codegen_inner_loops'):
                    code.splice(kernel.poststores)

            def get_reduction_code_buffer(loops, is_suffix=True):
                for loop in loops:
                    for kernel in loop.get_kernels():
                        if is_suffix:
                            return kernel.reduction_suffix
                        else:
                            return kernel.reduction_prefix
                return None

            def gen_loops(loops: List[LoopLevel], in_reduction=False):
                with contextlib.ExitStack() as stack_outer:
                    if loops:
                        loop = loops[0]
                        if loop.is_reduction() and (not in_reduction):
                            reduction_prefix = get_reduction_code_buffer(loops, is_suffix=False)
                            if reduction_prefix:
                                stack_outer.enter_context(code.indent())
                            code.splice(reduction_prefix)
                        if loop_nest.is_reduction_only() and loop.parallel:
                            worksharing.parallel(threads)
                    for loop in loops:
                        gen_loop(loop, in_reduction)
                    if loops:
                        loop = loops[0]
                        if loop_nest.is_reduction_only() and loop.parallel:
                            worksharing.close()
                        if loop.is_reduction() and (not in_reduction):
                            code.splice(get_reduction_code_buffer(loops, is_suffix=True))

            def gen_loop(loop: LoopLevel, in_reduction=False):
                with contextlib.ExitStack() as stack:
                    loop_lines = loop.lines()
                    if loop_lines is None:
                        return
                    code.writelines(loop_lines)
                    stack.enter_context(code.indent())
                    if loop.inner:
                        gen_loops(loop.inner, loop.is_reduction())
                    else:
                        kernels = loop.get_kernels()
                        assert len(kernels) == 1
                        gen_kernel(kernels[0])
            stack.enter_context(code.indent())
            if loop_nest.root:
                gen_loops(loop_nest.root)
            else:
                gen_kernel(loop_nest.kernel)

    def codegen_loops(self, code, worksharing):
        loop_nest = LoopNestWithSplit.build(self)
        self.codegen_loops_impl(loop_nest, code, worksharing)

    @property
    def assert_function(self) -> str:
        return 'TORCH_CHECK'

    def decide_parallel_depth(self, ranges, threads):
        seq = self.size_hint()
        par = 1
        depth = 0
        for expr in ranges:
            hint = V.graph.sizevars.size_hint(expr, fallback=8192)
            if par >= 2 * threads or par == threads:
                break
            if seq // threads < config.cpp.min_chunk_size:
                break
            depth += 1
            par *= hint
            seq /= hint
        if config.cpp.dynamic_threads and depth == 0 and (len(ranges) > 0):
            depth = 1
        return depth

    @contextlib.contextmanager
    def write_to_suffix(self):
        prior = (self.loads, self.compute, self.stores, self.cse)
        self.loads = IndentedBuffer()
        self.compute = IndentedBuffer()
        self.stores = IndentedBuffer()
        self.cse = self.cse.clone()
        yield
        self.reduction_suffix.splice(self.loads)
        self.reduction_suffix.splice(self.compute)
        self.reduction_suffix.splice(self.stores)
        self.loads, self.compute, self.stores, self.cse = prior

    def create_cse_var(self, *args, **kwargs):
        return CppCSEVariable(*args, **kwargs)