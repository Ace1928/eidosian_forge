from pathlib import Path
from typing import (
import torch
from torch import Tensor
from torch.optim import Optimizer
from typing_extensions import TypeAlias, overload
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
@runtime_checkable
class Steppable(Protocol):
    """To structurally type ``optimizer.step()``"""

    @overload
    def step(self, closure: None=...) -> None:
        ...

    @overload
    def step(self, closure: Callable[[], float]) -> float:
        ...

    def step(self, closure: Optional[Callable[[], float]]=...) -> Optional[float]:
        ...