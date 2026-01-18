import copy
from collections import defaultdict
import dataclasses
from typing import Dict, List, Optional, Tuple
import warnings
import sympy
import torch
import torch.fx
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.symbolic_shapes import SymInt
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo
from torch.utils._sympy.value_ranges import ValueRanges
from torch._export.passes.add_runtime_assertions_for_constraints_pass import (
from torch.export.graph_signature import (
from torch.export.exported_program import (
from .utils import _check_input_constraints_pre_hook
def _create_stateful_graph_module(plain_graph_module: torch.fx.GraphModule, range_constraints, equality_constraints):
    stateful_gm = _StatefulGraphModule._create(plain_graph_module, plain_graph_module.graph, range_constraints=range_constraints, equality_constraints=equality_constraints)
    stateful_gm.register_forward_pre_hook(_check_input_constraints_pre_hook, with_kwargs=True)
    return stateful_gm