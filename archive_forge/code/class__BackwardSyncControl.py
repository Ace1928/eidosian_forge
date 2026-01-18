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
class _BackwardSyncControl(ABC):
    """Interface for any :class:`Strategy` that wants to offer a functionality to enable or disable gradient
    synchronization during/after back-propagation.

    The most common use-case is gradient accumulation. If a :class:`Strategy` implements this interface, the user can
    implement their gradient accumulation loop very efficiently by disabling redundant gradient synchronization.

    """

    @abstractmethod
    def no_backward_sync(self, module: Module) -> ContextManager:
        """Blocks the synchronization of gradients during the backward pass.

        This is a context manager. It is only effective if it wraps a call to `.backward()`.

        """