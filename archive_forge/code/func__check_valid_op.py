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
def _check_valid_op(op) -> None:

    def _allowed_builtin_ops() -> List:
        ret = self.allowed_builtin_ops()
        assert all((inspect.isbuiltin(op) for op in ret))
        return ret

    def _allowed_op_types() -> Tuple[Type[Any], ...]:
        ret = self.allowed_op_types()
        assert not any((t is object for t in ret))
        return ret
    _allowed_torch_functions = (torch.autograd.grad_mode.set_grad_enabled,)
    if not isinstance(op, _allowed_op_types()):
        if op not in _allowed_builtin_ops() and op not in _allowed_torch_functions:
            raise SpecViolationError(f"Operator '{op}' is not an allowed operator type: {_allowed_op_types()}\nValid builtin ops: {_allowed_builtin_ops()}Valid torch functions: {_allowed_torch_functions}")
    if isinstance(op, OpOverload):
        if not is_functional(op):
            raise SpecViolationError(f"operator '{op}' is not functional")
    self.check_valid_op(op)