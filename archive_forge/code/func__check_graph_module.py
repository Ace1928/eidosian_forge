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
@final
def _check_graph_module(self, gm: torch.fx.GraphModule) -> None:

    def _allowed_getattr_types() -> Tuple[Type[Any], ...]:
        ret = self.allowed_getattr_types()
        assert not any((t is object for t in ret))
        return ret

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
    for mod in gm.modules():
        if not isinstance(mod, torch.fx.GraphModule):
            continue
        mod.graph.lint()
        for node in mod.graph.nodes:
            if node.op in {'call_module', 'call_method'}:
                raise SpecViolationError(f"call_module is not valid: got a class '{node.target}' ")
            elif node.op == 'call_function':
                _check_val(node)
                _check_valid_op(node.target)
            elif node.op == 'get_attr':
                if not isinstance(node.target, str):
                    raise SpecViolationError(f'Expected get_attr target to be string, but got {type(node.target)}')
                attr = getattr(mod, node.target)
                if isinstance(attr, torch.nn.Module):

                    def _is_type(name, ty):
                        return isinstance(getattr(attr, name, None), ty)
                    if type(attr).__name__ == 'LoweredBackendModule':
                        if _is_type('backend_id', str) and _is_type('processed_bytes', bytes) and _is_type('compile_specs', list) and hasattr(attr, 'original_module'):
                            continue
                        else:
                            backend_id = getattr(attr, 'backend_id', None)
                            processed_bytes = getattr(attr, 'processed_bytes', None)
                            compile_specs = getattr(attr, 'compile_specs', None)
                            raise SpecViolationError(f'Invalid get_attr type {type(attr)}. \nLoweredBackendModule fields: backend_id(str) : {type(backend_id)}, processed_bytes(bytes) : {type(processed_bytes)}, compile_specs(list) : {type(compile_specs)}')
                if not isinstance(attr, _allowed_getattr_types()):
                    raise SpecViolationError(f'Invalid get_attr type {type(attr)}. \nValid get_attr types: {_allowed_getattr_types()}')
            elif node.op == 'placeholder':
                _check_val(node)
    self.check_additional(gm)