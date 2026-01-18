import inspect
from collections import defaultdict
from functools import wraps
from itertools import chain
from typing import Callable, Dict, List, Sequence, Union
import torch
import torch.library
from torch._ops import HigherOrderOperator, OpOverload, OpOverloadPacket
from torch._prims_common import CustomOutParamAnnotation
from torch.utils import _pytree as pytree
import torch._decomp.decompositions
import torch._refs
def _add_op_to_registry(registry, op, fn):
    """
    This is an internal API for adding an op to the decomposition table.

    If op is OpOverload, it will be added to the registry directly.
    If op is OpOverloadPacket, all the valid op_overloads in the packet will be added to the registry.
    """
    overloads: List[Union[torch._ops.OperatorBase]] = []
    if isinstance(op, HigherOrderOperator):
        registry[op] = fn
        return
    elif isinstance(op, OpOverload):
        overloads.append(op)
    else:
        assert isinstance(op, OpOverloadPacket)
        for ol in op.overloads():
            overloads.append(getattr(op, ol))
    for op_overload in overloads:
        if op_overload in registry:
            raise RuntimeError(f'duplicate registrations for {op_overload}')
        if torch._C._dispatch_has_kernel(op_overload.name()):
            registry[op_overload] = fn