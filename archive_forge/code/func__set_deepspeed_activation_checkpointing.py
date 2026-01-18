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
def _set_deepspeed_activation_checkpointing(self) -> None:
    import deepspeed
    assert isinstance(self.config, dict)
    if self.config.get('activation_checkpointing'):
        checkpoint_config = self.config['activation_checkpointing']
        deepspeed.checkpointing.configure(mpu_=None, partition_activations=checkpoint_config.get('partition_activations'), contiguous_checkpointing=checkpoint_config.get('contiguous_memory_optimization'), checkpoint_in_cpu=checkpoint_config.get('cpu_checkpointing'), profile=checkpoint_config.get('profile'))