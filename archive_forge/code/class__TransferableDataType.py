from abc import ABC
from functools import partial
from typing import Any, Callable, List, Tuple, Union
import numpy as np
import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from lightning_fabric.utilities.types import _DEVICE
class _TransferableDataType(ABC):
    """A custom type for data that can be moved to a torch device via ``.to(...)``.

    Example:

        >>> isinstance(dict, _TransferableDataType)
        False
        >>> isinstance(torch.rand(2, 3), _TransferableDataType)
        True
        >>> class CustomObject:
        ...     def __init__(self):
        ...         self.x = torch.rand(2, 2)
        ...     def to(self, device):
        ...         self.x = self.x.to(device)
        ...         return self
        >>> isinstance(CustomObject(), _TransferableDataType)
        True

    """

    @classmethod
    def __subclasshook__(cls, subclass: Any) -> Union[bool, Any]:
        if cls is _TransferableDataType:
            to = getattr(subclass, 'to', None)
            return callable(to)
        return NotImplemented