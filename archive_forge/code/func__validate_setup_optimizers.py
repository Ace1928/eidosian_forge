import inspect
import os
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path
from typing import (
import torch
import torch.nn as nn
from lightning_utilities.core.apply_func import apply_to_collection
from lightning_utilities.core.overrides import is_overridden
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from lightning_fabric.accelerators.accelerator import Accelerator
from lightning_fabric.connector import _PLUGIN_INPUT, _PRECISION_INPUT, _Connector, _is_using_cli
from lightning_fabric.loggers import Logger
from lightning_fabric.plugins import Precision  # avoid circular imports: # isort: split
from lightning_fabric.strategies import (
from lightning_fabric.strategies.fsdp import _has_meta_device_parameters
from lightning_fabric.strategies.launchers import _MultiProcessingLauncher, _XLALauncher
from lightning_fabric.strategies.strategy import TBroadcast, _Sharded
from lightning_fabric.utilities import move_data_to_device
from lightning_fabric.utilities.apply_func import convert_tensors_to_scalars, convert_to_tensors
from lightning_fabric.utilities.data import (
from lightning_fabric.utilities.device_dtype_mixin import _update_properties
from lightning_fabric.utilities.distributed import DistributedSamplerWrapper
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from lightning_fabric.utilities.rank_zero import rank_zero_deprecation, rank_zero_warn
from lightning_fabric.utilities.registry import _load_external_callbacks
from lightning_fabric.utilities.seed import seed_everything
from lightning_fabric.utilities.types import ReduceOp
from lightning_fabric.utilities.warnings import PossibleUserWarning
from lightning_fabric.wrappers import (
def _validate_setup_optimizers(self, optimizers: Sequence[Optimizer]) -> None:
    self._validate_launched()
    if isinstance(self._strategy, (DeepSpeedStrategy, XLAStrategy)):
        raise RuntimeError(f'The `{type(self._strategy).__name__}` requires the model and optimizer(s) to be set up jointly through `.setup(model, optimizer, ...)`.')
    if not optimizers:
        raise ValueError('`setup_optimizers` requires at least one optimizer as input.')
    if any((isinstance(opt, _FabricOptimizer) for opt in optimizers)):
        raise ValueError('An optimizer should be passed only once to the `setup_optimizers` method.')
    if any((_has_meta_device_parameters(optimizer) for optimizer in optimizers)):
        raise RuntimeError("The optimizer has references to the model's meta-device parameters. Materializing them is is currently not supported. Create the optimizer after setting up the model, then call `fabric.setup_optimizers(optimizer)`.")