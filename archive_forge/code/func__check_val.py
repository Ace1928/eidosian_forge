import inspect
import math
import operator
from collections.abc import Iterable
from typing import Any, Dict, final, List, Optional, Tuple, Type
import torch
from torch._ops import HigherOrderOperator, OpOverload
from torch._subclasses.fake_tensor import FakeTensor
from torch.export.exported_program import ExportedProgram
from torch.export.graph_signature import (
from torch.fx import GraphModule
from torch.fx.experimental.symbolic_shapes import SymBool, SymFloat, SymInt
def _check_val(node: torch.fx.Node) -> None:

    def _check_correct_val(val):
        if val is None:
            return True
        elif isinstance(val, (int, bool, str, float)):
            return True
        elif isinstance(val, (torch.memory_format, torch.dtype, torch.device, torch.layout)):
            return True
        elif isinstance(val, (FakeTensor, torch.Tensor)):
            return True
        elif isinstance(val, (SymInt, SymFloat, SymBool)):
            return True
        elif isinstance(val, Iterable):
            return all((_check_correct_val(x) for x in val))
        return False

    def _no_returns(op):
        if not isinstance(op, OpOverload):
            return False
        return len(op._schema.returns) == 0
    if 'val' not in node.meta:
        if node.op == 'call_function' and _no_returns(node.target):
            return
        raise SpecViolationError(f'Node.meta {node.name} is missing val field.')
    val = node.meta['val']
    if not _check_correct_val(val):
        raise SpecViolationError(f'Node.meta {node.name} has invalid val field {val}')