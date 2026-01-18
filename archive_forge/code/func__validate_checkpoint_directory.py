import argparse
import json
import logging
import os
import platform
from contextlib import ExitStack
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Dict, List, Mapping, Optional, Tuple, Union
import torch
from lightning_utilities.core.imports import RequirementCache
from torch.nn import Module
from torch.optim import Optimizer
from typing_extensions import override
from lightning_fabric.accelerators import Accelerator, CUDAAccelerator
from lightning_fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning_fabric.plugins.precision import Precision
from lightning_fabric.strategies.ddp import DDPStrategy
from lightning_fabric.strategies.registry import _StrategyRegistry
from lightning_fabric.strategies.strategy import _Sharded
from lightning_fabric.utilities.distributed import log
from lightning_fabric.utilities.load import _move_state_into
from lightning_fabric.utilities.rank_zero import rank_zero_info, rank_zero_warn
from lightning_fabric.utilities.seed import reset_seed
from lightning_fabric.utilities.types import _PATH
def _validate_checkpoint_directory(path: _PATH) -> None:
    """Validates that the path points to a DeepSpeed checkpoint directory and suggests fixes for user error."""
    path = Path(path)
    path_is_ds_checkpoint = _is_deepspeed_checkpoint(path)
    default_message = f'The provided path is not a valid DeepSpeed checkpoint: {path}'
    if not path_is_ds_checkpoint:
        parent_is_ds_checkpoint = _is_deepspeed_checkpoint(path.parent)
        if parent_is_ds_checkpoint:
            raise FileNotFoundError(f'{default_message}. It looks like you passed the path to a subfolder. Try to load using this parent directory instead: {path.parent}')
        parent_parent_is_ds_checkpoint = path.is_file() and _is_deepspeed_checkpoint(path.parent.parent)
        if parent_parent_is_ds_checkpoint:
            raise FileNotFoundError(f'{default_message}. It looks like you passed the path to a file inside a DeepSpeed checkpoint folder. Try to load using this parent directory instead: {path.parent.parent}')
        raise FileNotFoundError(default_message)