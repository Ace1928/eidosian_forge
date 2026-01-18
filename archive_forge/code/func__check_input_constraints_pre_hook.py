import dataclasses
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type
import torch
from torch._export import ExportedProgram
from torch.utils._pytree import (
@torch._dynamo.disable
def _check_input_constraints_pre_hook(self, *args, **kwargs):
    flat_args, _ = tree_flatten(args)
    return _check_input_constraints_for_graph(self.graph, range_constraints=self.range_constraints, equality_constraints=self.equality_constraints)(*flat_args)