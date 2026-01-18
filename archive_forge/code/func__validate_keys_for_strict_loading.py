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
def _validate_keys_for_strict_loading(requested_keys: Iterable[str], checkpoint_keys: Iterable[str], strict: bool) -> None:
    invalid_keys = [k for k in requested_keys if k not in checkpoint_keys]
    if strict and invalid_keys:
        raise KeyError(f"The requested state contains a key '{invalid_keys[0]}' that does not exist in the loaded checkpoint. To disable strict loading, set `strict=False`.")