from __future__ import annotations
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union
import torch
from torch._C import DisableTorchFunctionSubclass
from torch.types import _device, _dtype, _size
from torchvision.tv_tensors._torch_function_helpers import _FORCE_TORCHFUNCTION_SUBCLASS, _must_return_subclass
@classmethod
def _wrap_output(cls, output: torch.Tensor, args: Sequence[Any]=(), kwargs: Optional[Mapping[str, Any]]=None) -> torch.Tensor:
    if isinstance(output, torch.Tensor) and (not isinstance(output, cls)):
        output = output.as_subclass(cls)
    if isinstance(output, (tuple, list)):
        output = type(output)((cls._wrap_output(part, args, kwargs) for part in output))
    return output