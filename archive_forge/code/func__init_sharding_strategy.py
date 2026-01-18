import shutil
from contextlib import ExitStack
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import (
import torch
from lightning_utilities.core.imports import RequirementCache
from lightning_utilities.core.rank_zero import rank_zero_only as utils_rank_zero_only
from torch import Tensor
from torch.nn import Module, Parameter
from torch.optim import Optimizer
from typing_extensions import TypeGuard, override
from lightning_fabric.accelerators import Accelerator
from lightning_fabric.plugins import CheckpointIO, ClusterEnvironment, Precision
from lightning_fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning_fabric.plugins.precision.fsdp import FSDPPrecision
from lightning_fabric.strategies.launchers.subprocess_script import _SubprocessScriptLauncher
from lightning_fabric.strategies.parallel import ParallelStrategy
from lightning_fabric.strategies.registry import _StrategyRegistry
from lightning_fabric.strategies.strategy import (
from lightning_fabric.utilities.distributed import (
from lightning_fabric.utilities.distributed import group as _group
from lightning_fabric.utilities.imports import (
from lightning_fabric.utilities.init import _EmptyInit
from lightning_fabric.utilities.load import _METADATA_FILENAME, _lazy_load, _materialize_tensors, _move_state_into
from lightning_fabric.utilities.rank_zero import rank_zero_deprecation, rank_zero_only, rank_zero_warn
from lightning_fabric.utilities.seed import reset_seed
from lightning_fabric.utilities.types import _PATH, _Stateful
def _init_sharding_strategy(sharding_strategy: '_SHARDING_STRATEGY', kwargs: Dict) -> 'ShardingStrategy':
    from torch.distributed.fsdp import ShardingStrategy
    if kwargs.get('process_group') is not None and kwargs.get('device_mesh') is not None:
        raise ValueError('The arguments `FSDPStrategy(process_group=..., device_mesh=...)` are mutually exclusive.Pass only one of them.')
    strategy = ShardingStrategy[sharding_strategy.upper()] if isinstance(sharding_strategy, str) else sharding_strategy
    if 'HYBRID' in strategy.name and kwargs.get('auto_wrap_policy') is None and (kwargs.get('process_group') is None) and (kwargs.get('device_mesh') is None):
        raise RuntimeError('The hybrid sharding strategy requires you to pass at least one of the parameters: `auto_wrap_policy`, `process_group` tuple, or `device_mesh`.')
    return strategy