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
class _StatefulGraphModule(torch.fx.GraphModule, metaclass=_StatefulGraphModuleFactory):

    def __init__(self, root, graph, range_constraints=None, equality_constraints=None):
        super().__init__(root, graph)
        self.range_constraints = range_constraints or []
        self.equality_constraints = equality_constraints or []