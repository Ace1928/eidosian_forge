from __future__ import annotations
from ..language import core as lcore
from . import torch_wrapper
from .core import ExecutionContext
from .memory_map import MemoryMap
def _infer_tensor(func):
    """
    A decorator function to harmonize function args:
        - converts primitives to PyTorch tensors
        - wraps PyTorch tensors with WrappedTensors
    """

    def wrapper(*args):
        new_args = tuple(map(lambda v: _primitive_to_tensor(v), args))
        new_args = tuple(map(lambda v: WrappedTensor(v) if torch.is_tensor(v) else v, new_args))
        return func(*new_args)
    return wrapper