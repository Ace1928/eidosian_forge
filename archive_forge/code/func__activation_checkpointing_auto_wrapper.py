import io
from contextlib import ExitStack, nullcontext
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Dict, List, Literal, Optional, Set, Tuple, Type, Union
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing_extensions import override
from lightning_fabric.accelerators import Accelerator
from lightning_fabric.accelerators.xla import _XLA_AVAILABLE
from lightning_fabric.plugins import XLAPrecision
from lightning_fabric.plugins.environments import XLAEnvironment
from lightning_fabric.plugins.io.xla import XLACheckpointIO
from lightning_fabric.strategies import ParallelStrategy, _StrategyRegistry
from lightning_fabric.strategies.fsdp import _apply_filter
from lightning_fabric.strategies.launchers.xla import _XLALauncher
from lightning_fabric.strategies.strategy import (
from lightning_fabric.utilities.cloud_io import get_filesystem
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from lightning_fabric.utilities.init import _EmptyInit
from lightning_fabric.utilities.rank_zero import rank_zero_only, rank_zero_warn
from lightning_fabric.utilities.types import _PATH, Optimizable, ReduceOp
def _activation_checkpointing_auto_wrapper(policy: _POLICY_SET, module: Module, *args: Any, **kwargs: Any) -> Module:
    from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as XLAFSDP
    from torch_xla.distributed.fsdp import checkpoint_module
    module = checkpoint_module(module) if isinstance(module, tuple(policy)) else module
    return XLAFSDP(module, *args, **kwargs)