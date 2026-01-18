from __future__ import annotations
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union
import torch
from torch._C import DisableTorchFunctionSubclass
from torch.types import _device, _dtype, _size
from torchvision.tv_tensors._torch_function_helpers import _FORCE_TORCHFUNCTION_SUBCLASS, _must_return_subclass
def _make_repr(self, **kwargs: Any) -> str:
    extra_repr = ', '.join((f'{key}={value}' for key, value in kwargs.items()))
    return f'{super().__repr__()[:-1]}, {extra_repr})'