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
def _set_node_environment_variables(self) -> None:
    assert self.cluster_environment is not None
    os.environ['MASTER_ADDR'] = self.cluster_environment.main_address
    os.environ['MASTER_PORT'] = str(self.cluster_environment.main_port)
    os.environ['RANK'] = str(self.global_rank)
    os.environ['WORLD_SIZE'] = str(self.world_size)
    os.environ['LOCAL_RANK'] = str(self.local_rank)