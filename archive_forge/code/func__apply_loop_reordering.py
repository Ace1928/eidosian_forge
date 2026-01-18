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
def _apply_loop_reordering(index_vars, support_vars, sizes, memory_addrs, reordering_reindex=None, priority_idx=None):
    """
        Shuffle the order of loops around to hopefully improve performance.
        """
    from .scheduler import pick_loop_order
    if priority_idx is None:
        priority_idx = []
    try:
        strides = [V.graph.sizevars.stride_hints(expr, index_vars, support_vars) for expr in memory_addrs]
        assert len(strides) == len(memory_addrs) and len(strides[0]) == len(index_vars)
        if reordering_reindex is not None:
            for i in range(len(memory_addrs)):
                try:
                    strides[i] = reordering_reindex[i](strides[i])
                except AssertionError:
                    pass
        order = list(reversed(pick_loop_order(strides, sizes, priority_idx)))
    except Exception:
        if config.debug:
            log.warning('Did not simplify complex index:\n%s\n%s', dict(zip(index_vars, sizes)), memory_addrs)
        order = list(range(len(sizes)))
    sizes = [sizes[i] for i in order]
    return (sizes, same_reorder(order), inverse_reorder(order))