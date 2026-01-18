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
@staticmethod
def _unroll_reduction_fn(inner_fn, reduction_ranges, reduction_type, src_dtype):
    """Convert inner_fn from a reduction to an pointwise"""
    reduction_ranges = [V.graph.sizevars.evaluate_static_shape(x) for x in reduction_ranges]
    combine_fn = get_reduction_combine_fn(reduction_type, src_dtype)

    def fn(index):
        return functools.reduce(combine_fn, (value_fn(index, rindex) for rindex in itertools.product(*[range(x) for x in reduction_ranges])))
    if reduction_type in ('argmin', 'argmax'):
        flatten_index = FixedLayout(None, None, reduction_ranges, FlexibleLayout.contiguous_strides(reduction_ranges)).make_indexer()

        def value_fn(index, rindex):
            rindex = [sympy.expand(i) for i in rindex]
            return (inner_fn(index, rindex), ops.index_expr(flatten_index(rindex), torch.int64))
        return lambda index: fn(index)[1]
    else:
        value_fn = inner_fn
        return fn