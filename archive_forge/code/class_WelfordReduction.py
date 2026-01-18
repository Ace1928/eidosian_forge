import collections
import contextlib
import dataclasses
import functools
import itertools
import logging
import re
import textwrap
import traceback
from contextlib import nullcontext
from enum import Enum
from functools import partial
from inspect import signature
from typing import (
from unittest.mock import patch
import sympy
from sympy import Expr, Integer
import torch._export.serde.schema as export_schema
import torch._logging
import torch.fx
import torch.utils._pytree as pytree
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import identity
from torch._export.serde.serialize import GraphModuleSerializer
from torch._prims_common import (
from torch._subclasses.fake_tensor import get_schema_info
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols, SymTypes
from torch.utils._sympy.functions import CleanDiv, FloorDiv, ModularIndexing
from . import config, dependencies
from .codegen.common import index_prevent_reordering
from .dependencies import (
from .utils import (
from .virtualized import ops, V
class WelfordReduction(Reduction):
    output_index: int

    def __init__(self, device, dtype, inner_fns, ranges, reduction_ranges, reduction_type, reduction_hint, output_index):
        if len(inner_fns) == 1:
            loader = inner_fns[0]
        else:

            def loader(idx, reduction_idx):
                return tuple((fn(idx, reduction_idx) for fn in inner_fns))
        super().__init__(device, dtype, loader, ranges, reduction_ranges, reduction_type, dtype, reduction_hint)
        self.output_index = output_index

    def store_reduction(self, output_name, indexer, vars, reduction_vars):
        values = ops.reduction(self.dtype, self.src_dtype, self.reduction_type, self.inner_fn(vars, reduction_vars))
        value = values[self.output_index]
        return ops.store_reduction(output_name, indexer(vars), value)

    @classmethod
    def create(cls, device: torch.device, dtype: torch.dtype, inner_fns: Sequence[Callable[..., Any]], ranges: List[Expr], reduction_ranges: List[Expr], reduction_type: str, reduction_hint: ReductionHint=ReductionHint.DEFAULT):
        assert reduction_type in {'welford_reduce', 'welford_combine'}
        reduction_numel = V.graph.sizevars.simplify(sympy_product(reduction_ranges))

        def const(val):

            def inner_fn(idx):
                return ops.constant(val, dtype)
            return Pointwise.create(device=device, dtype=dtype, inner_fn=inner_fn, ranges=list(ranges))
        if reduction_numel == 0:
            mean = const(0)
            m2 = const(0)
            weight = const(0)
            return (mean, m2, weight)
        if reduction_numel == 1:

            def copy(loader):

                def inner_fn(idx):
                    reduction_index = [sympy.Integer(0) for _ in reduction_ranges]
                    return loader(idx, reduction_index)
                return Pointwise.create(device=device, dtype=dtype, inner_fn=inner_fn, ranges=list(ranges))
            if reduction_type == 'welford_reduce':
                return (copy(inner_fns[0]), const(0), const(1))
            else:
                return tuple((copy(fn) for fn in inner_fns))
        hint, split = Reduction.num_splits(device, dtype, dtype, inner_fns[0], ranges, reduction_ranges, reduction_type=reduction_type, reduction_numel=reduction_numel)
        if reduction_hint == ReductionHint.DEFAULT:
            reduction_hint = hint
        if split > 1:
            return cls.create_multilayer(device, dtype, inner_fns, ranges, reduction_ranges, reduction_type, split, reduction_hint)
        results = [TensorBox.create(WelfordReduction(device, dtype, inner_fns, ranges, reduction_ranges, reduction_type, reduction_hint, output_idx)) for output_idx in range(3)]
        for t in results:
            t.realize()
        return results

    @staticmethod
    def default_value(reduction_type, dtype):
        return (0, 0, 0)

    @classmethod
    def create_multilayer(cls, device: torch.device, dtype: torch.dtype, inner_fns: Sequence[Callable[..., Any]], ranges: List[Expr], reduction_ranges: List[Expr], reduction_type: str, split: int, reduction_hint: ReductionHint):
        """
        Break a large reduction up into multiple smaller reductions
        recursively
        """
        reduction_numel = sympy_product(reduction_ranges)
        need_mask = not V.graph.sizevars.is_expr_static_and_true(sympy.Eq(reduction_numel % split, 0))
        if need_mask and reduction_type != 'welford_combine':

            def constant(idx, reduction_idx, value):
                return ops.constant(value, dtype)
            return cls.create_multilayer(device=device, dtype=dtype, inner_fns=(inner_fns[0], partial(constant, value=0), partial(constant, value=1)), ranges=ranges, reduction_ranges=reduction_ranges, reduction_type='welford_combine', split=split, reduction_hint=reduction_hint)
        block_size = FloorDiv(reduction_numel + (split - 1), split)
        intermediates = WelfordReduction.create(device, dtype, tuple((cls._multilayer_wrap_loader(loader, reduction_ranges, reduction_numel, split, block_size, default=0) for loader in inner_fns)), [*ranges, split], [block_size], reduction_type, reduction_hint)
        for i in intermediates:
            i.realize()
        i_loaders = [i.make_loader() for i in intermediates]

        def intermediate_loader_fn(index, reduction_index, loader):
            return loader([*index, *reduction_index])
        numel_hint = V.graph.sizevars.size_hint(sympy_product(ranges))
        reduction_hint = cls._multilayer_second_step_hint(split, numel_hint, reduction_hint)
        return WelfordReduction.create(device, dtype, tuple((partial(intermediate_loader_fn, loader=i.make_loader()) for i in intermediates)), ranges, [split], 'welford_combine', reduction_hint)