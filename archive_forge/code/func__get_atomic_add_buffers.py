import collections
import dataclasses
import functools
import itertools
import logging
import math
import os
import pprint
import textwrap
from typing import (
import sympy
import torch
from torch._dynamo.utils import dynamo_timed
from torch._inductor.metrics import get_metric_table, is_metric_table_enabled
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.utils._triton import has_triton
from . import comms, config, dependencies, ir, metrics
from .codegen.common import get_scheduling_for_device, Kernel
from .comm_analysis import estimate_nccl_collective_runtime
from .dependencies import StarDep, WeakDep
from .ir import ComputedBuffer, MultiOutput, MultiOutputLayout
from .sizevars import SimplifyIndexing
from .utils import (
from .virtualized import V
@cache_on_self
def _get_atomic_add_buffers(self) -> Set[str]:
    buffers_store_as_atomic_add = set()
    if isinstance(self._body, ir.LoopBody):
        for node in self._body.get_nodes():
            if node.op == 'call_method' and node.target == 'store' and ('mode' in node.kwargs and node.kwargs['mode'] == 'atomic_add' or (len(node.args) == 5 and node.args[4] == 'atomic_add')):
                buffers_store_as_atomic_add.add(node.kwargs['name'] if 'name' in node.kwargs else node.args[1] if len(node.args) >= 2 else '')
    return buffers_store_as_atomic_add