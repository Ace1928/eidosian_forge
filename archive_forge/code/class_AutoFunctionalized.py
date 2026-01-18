from typing import Any, Dict, List, Tuple
import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch._prims_common import clone_preserve_strides
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
class AutoFunctionalized(HigherOrderOperator):
    """auto_functionalized(op, mutated_args_names, kwargs)

    This HOP runs a "functional" version of op.

    Concretely, it clones kwargs that `op` mutates (specified by
    mutated_args_names), runs `op(**kwargs)` with the cloned values,
    and then returns a flat Tuple of the cloned values that were mutated.

    We have some restrictions on `op`, most notably that it returns None.
    See `can_auto_functionalize` for the restrictions. We can likely lift
    many of these if users request it.
    """

    def __init__(self):
        super().__init__('auto_functionalized')

    def __call__(self, op: torch._ops.OpOverload, mutated_args_names: List[str], kwargs: Dict[str, Any]) -> Tuple[Tensor, ...]:
        assert can_auto_functionalize(op)
        assert isinstance(mutated_args_names, list)
        assert isinstance(kwargs, dict)
        return super().__call__(op, mutated_args_names, kwargs)