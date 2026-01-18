import itertools
from typing import Any, Callable, Dict, Optional, Sequence
import torch
from torch.overrides import TorchFunctionMode
from typing_extensions import override
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_1
from lightning_fabric.utilities.types import _DEVICE
class _EmptyInit(TorchFunctionMode):
    """Initialize `nn.Module` with empty tensors, i.e., uninitialized memory.

    Example::

        with _EmptyInit():
            model = BigModel()
        model.load_state_dict(torch.load("checkpoint.pt"))

    """

    def __init__(self, enabled: bool=True) -> None:
        super().__init__()
        self.enabled = enabled

    @override
    def __torch_function__(self, func: Callable, types: Sequence, args: Sequence[Any]=(), kwargs: Optional[Dict]=None) -> Any:
        kwargs = kwargs or {}
        if not self.enabled:
            return func(*args, **kwargs)
        if getattr(func, '__module__', None) == 'torch.nn.init':
            if 'tensor' in kwargs:
                return kwargs['tensor']
            return args[0]
        return func(*args, **kwargs)