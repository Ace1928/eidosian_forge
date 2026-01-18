from __future__ import annotations
from ..language import core as lcore
from . import torch_wrapper
from .core import ExecutionContext
from .memory_map import MemoryMap
def _tensor_operation(func):
    """
    A decorator function to unwrap WrappedTensors and debugger_constexpr before calling the function.
    Can be combined with _infer_tensor decorator to harmonize args (everything to torch tensor).
    """

    def wrapper(*args, **kwargs):
        for arg in args:
            assert not torch.is_tensor(arg), 'unexpected tensor argument'

        def unwrap_tensor(v):
            if isinstance(v, WrappedTensor):
                return v.tensor
            if isinstance(v, debugger_constexpr):
                return v.value
            return v
        new_args = tuple(map(unwrap_tensor, args))
        new_kwargs = {k: unwrap_tensor(v) for k, v in kwargs.items()}
        result = func(args[0], *new_args[1:], **new_kwargs)
        return WrappedTensor(result) if torch.is_tensor(result) else result
    return wrapper