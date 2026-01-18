from __future__ import annotations
from ..language import core as lcore
from . import torch_wrapper
from .core import ExecutionContext
from .memory_map import MemoryMap
def _primitive_to_tensor(x):
    """
    Converts various Python primitive data types to PyTorch tensor.
    """
    tensor_args = {'device': 'cuda'}
    if isinstance(x, bool):
        return torch.tensor([x], dtype=torch.bool, **tensor_args)
    elif isinstance(x, int):
        if -2 ** 31 <= x < 2 ** 31:
            return torch.tensor([x], dtype=torch.int32, **tensor_args)
        elif -2 ** 63 <= x < 2 ** 63:
            return torch.tensor([x], dtype=torch.int64, **tensor_args)
        else:
            raise RuntimeError(f'Nonrepresentable integer {x}.')
    elif isinstance(x, float):
        return torch.tensor([x], dtype=torch.float32, **tensor_args)
    elif torch.is_tensor(x):
        return x
    elif isinstance(x, WrappedTensor):
        return x
    elif isinstance(x, debugger_constexpr):
        if x.value is None:
            return None
        return _primitive_to_tensor(x.value)
    elif x is None:
        return None
    assert False, f'cannot convert {x} of type {type(x)} to tensor'