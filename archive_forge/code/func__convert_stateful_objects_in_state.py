import logging
from abc import ABC, abstractmethod
from contextlib import ExitStack
from typing import Any, Callable, ContextManager, Dict, Iterable, List, Optional, Tuple, TypeVar, Union
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from lightning_fabric.accelerators import Accelerator
from lightning_fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning_fabric.plugins.io.torch_io import TorchCheckpointIO
from lightning_fabric.plugins.precision import Precision
from lightning_fabric.strategies.launchers.launcher import _Launcher
from lightning_fabric.strategies.registry import _StrategyRegistry
from lightning_fabric.utilities.apply_func import move_data_to_device
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from lightning_fabric.utilities.init import _EmptyInit
from lightning_fabric.utilities.types import _PATH, Optimizable, ReduceOp, _Stateful
def _convert_stateful_objects_in_state(self, state: Dict[str, Union[Module, Optimizer, Any]], filter: Dict[str, Callable[[str, Any], bool]]) -> Dict[str, Any]:
    converted_state: Dict[str, Any] = {}
    for key, obj in state.items():
        if isinstance(obj, Module):
            converted = self.get_module_state_dict(module=obj)
        elif isinstance(obj, Optimizer):
            converted = self.get_optimizer_state(optimizer=obj)
        elif isinstance(obj, _Stateful):
            converted = obj.state_dict()
        else:
            converted = obj
        _apply_filter(key, filter, converted, converted_state)
    return converted_state