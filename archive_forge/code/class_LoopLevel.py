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
@dataclasses.dataclass
class LoopLevel:
    var: Optional[sympy.Expr] = None
    size: Optional[sympy.Expr] = None
    offset: sympy.Expr = sympy.Integer(0)
    steps: sympy.Expr = sympy.Integer(1)
    parallel: int = 0
    simd_omp: bool = False
    simd_vec: bool = False
    collapsed: bool = False
    reduction_var_map: Optional[Dict[str, str]] = None
    parent: Optional['LoopLevel'] = None
    inner: List['LoopLevel'] = dataclasses.field(default_factory=list)
    kernel: Optional[CppKernel] = None

    def __post_init__(self):
        picked_vec_isa: codecache.VecISA = codecache.pick_vec_isa()
        self.simd_nelements: int = picked_vec_isa.nelements() if picked_vec_isa else 0

    def get_kernels(self) -> List[CppKernel]:
        """Get all kernel objects under this loop level"""
        if self.kernel:
            return [self.kernel]
        kernels = []
        for loop in self.inner:
            kernels += loop.get_kernels()
        return kernels

    def set_kernel(self, kernel: CppKernel):
        """
        Set the kernel under this loop level. No split is allowed under
        this loop level.
        """
        if not self.inner:
            self.kernel = kernel
            loop: Optional[LoopLevel] = self
            assert loop is not None
            if loop.is_reduction():
                loop.reduction_var_map = kernel.reduction_var_map.copy()
                loop = loop.parent
                while loop is not None and loop.is_reduction():
                    assert loop.reduction_var_map is not None
                    loop.reduction_var_map.update(kernel.reduction_var_map)
                    loop = loop.parent
            return
        assert len(self.inner) == 1
        self.inner[0].set_kernel(kernel)

    def get_loops_at(self, depth) -> List['LoopLevel']:
        if depth == 0:
            return [self]
        else:
            loops = []
            for loop in self.inner:
                loops += loop.get_loops_at(depth - 1)
            return loops

    def is_reduction(self):
        return bool(self.reduction_var_map)

    def split_with_tiling(self, depth, factor):

        def clone_inner():
            inner = []
            if self.inner:
                for loop in self.inner:
                    inner.append(loop.clone())
            return inner

        def do_split_with_tiling():
            sympy_factor = sympy.Integer(factor)
            offset = FloorDiv(self.size, sympy_factor) * sympy_factor
            main_loop = LoopLevel(self.var, offset)
            main_loop.steps = sympy_factor
            main_loop.parallel = self.parallel
            main_loop.collapsed = False
            main_loop.reduction_var_map = self.reduction_var_map
            main_loop.inner = clone_inner()
            if main_loop.inner:
                for loop in main_loop.inner:
                    loop.parent = main_loop
            tail_loop = LoopLevel(self.var, self.size)
            tail_loop.offset = offset
            tail_loop.parallel = self.parallel
            tail_loop.collapsed = False
            tail_loop.reduction_var_map = self.reduction_var_map
            tail_loop.inner = clone_inner()
            if tail_loop.inner:
                for loop in tail_loop.inner:
                    loop.parent = tail_loop
            return (main_loop, tail_loop)
        if depth == 0:
            main_loop, tail_loop = do_split_with_tiling()
            parent = self.parent
            if parent:
                parent.inner = [main_loop, tail_loop]
                main_loop.parent = parent
                tail_loop.parent = parent
            return (main_loop, tail_loop)
        else:
            assert len(self.inner) == 1
            return self.inner[0].split_with_tiling(depth - 1, factor)

    def clone(self):
        loop = copy(self)
        loop.inner = []
        if self.inner:
            for inner_loop in self.inner:
                inner_loop_clone = inner_loop.clone()
                inner_loop_clone.parent = loop
                loop.inner.append(inner_loop_clone)
        loop.kernel = deepcopy(self.kernel)
        return loop

    def lines(self):
        offset_expr = cexpr_index(self.offset)
        size_expr = cexpr_index(self.size)
        if config.cpp.no_redundant_loops and offset_expr == size_expr:
            return None
        if self.reduction_var_map:
            reduction = ' ' + ' '.join((f'reduction({RTYPE_TO_CPP[rtype]}:{var})' for var, rtype in self.reduction_var_map.items()))
        else:
            reduction = ''
        simd = f'simd simdlen({self.simd_nelements}) ' if self.simd_omp and self.simd_nelements > 1 else ''
        if self.parallel:
            line1 = f'#pragma omp for{reduction} '
            if self.parallel > 1:
                line1 += f' collapse({self.parallel})'
            if self.simd_omp:
                line1 = line1.replace(' for ', f' for {simd}')
        elif self.simd_vec:
            line1 = ''
        elif self.simd_omp:
            line1 = f'#pragma omp {simd}{reduction}'
        elif not self.reduction_var_map and codecache.is_gcc():
            line1 = '#pragma GCC ivdep'
        else:
            line1 = ''
        offset_str = f'{INDEX_TYPE} {self.var}={offset_expr}'
        size_str = f'{self.var}<{size_expr}'
        steps_str = f'{self.var}+={cexpr_index(self.steps)}'
        line2 = f'for({offset_str}; {size_str}; {steps_str})'
        if self.collapsed or not line1:
            return [line2]
        return [line1, line2]