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
class TritonCSEVariable(CSEVariable):

    def __init__(self, name, bounds: ValueRanges):
        super().__init__(name, bounds)
        self.mask_vars: Set[str] = set()

    def update_on_args(self, name, args, kwargs):
        if name == 'where':
            return
        for arg in args:
            if isinstance(arg, TritonCSEVariable):
                self.mask_vars.update(arg.mask_vars)
            elif isinstance(arg, sympy.Symbol) and arg.name[0] in 'xyr':
                self.mask_vars.update({f'{arg.name[0]}mask'})