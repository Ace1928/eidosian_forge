from pathlib import Path
from typing import (
import torch
from torch import Tensor
from torch.optim import Optimizer
from typing_extensions import TypeAlias, overload
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
@runtime_checkable
class Optimizable(Steppable, Protocol):
    """To structurally type ``optimizer``"""
    param_groups: List[Dict[Any, Any]]
    defaults: Dict[Any, Any]
    state: DefaultDict[Tensor, Any]

    def state_dict(self) -> Dict[str, Dict[Any, Any]]:
        ...

    def load_state_dict(self, state_dict: Dict[str, Dict[Any, Any]]) -> None:
        ...