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