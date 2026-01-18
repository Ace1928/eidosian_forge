from pathlib import Path
from typing import (
import torch
from torch import Tensor
from torch.optim import Optimizer
from typing_extensions import TypeAlias, overload
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
@runtime_checkable
class CollectibleGroup(Protocol):

    def size(self) -> int:
        ...

    def rank(self) -> int:
        ...