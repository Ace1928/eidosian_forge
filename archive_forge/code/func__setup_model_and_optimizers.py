import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Dict, Generator, List, Mapping, Optional, Tuple, TypeVar, Union
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
import pytorch_lightning as pl
from lightning_fabric.plugins import CheckpointIO
from lightning_fabric.strategies import _StrategyRegistry
from lightning_fabric.utilities import move_data_to_device
from lightning_fabric.utilities.distributed import ReduceOp
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from lightning_fabric.utilities.init import _EmptyInit
from lightning_fabric.utilities.optimizer import _optimizer_to_device, _optimizers_to_device
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.core.optimizer import LightningOptimizer, _init_optimizers_and_lr_schedulers
from pytorch_lightning.plugins import TorchCheckpointIO
from pytorch_lightning.plugins.io.wrapper import _WrappingCheckpointIO
from pytorch_lightning.plugins.precision import Precision
from pytorch_lightning.strategies.launchers.launcher import _Launcher
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.types import STEP_OUTPUT, LRSchedulerConfig
def _setup_model_and_optimizers(self, model: Module, optimizers: List[Optimizer]) -> Tuple[Module, List[Optimizer]]:
    """Setup a model and multiple optimizers together.

        The returned objects are expected to be in the same order they were passed in. The default implementation will
        call :meth:`_setup_model` and :meth:`_setup_optimizer` on the inputs.

        """
    model = self._setup_model(model)
    optimizers = [self._setup_optimizer(optimizer) for optimizer in optimizers]
    return (model, optimizers)