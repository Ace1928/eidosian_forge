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
def _initialize_engine(self, model: Module, optimizer: Optional[Optimizer]=None) -> Tuple['DeepSpeedEngine', Optimizer]:
    """Initialize one model and one optimizer with an optional learning rate scheduler.

        This calls :func:`deepspeed.initialize` internally.

        """
    import deepspeed
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    deepspeed_engine, deepspeed_optimizer, _, _ = deepspeed.initialize(args=argparse.Namespace(device_rank=self.root_device.index), config=self.config, model=model, model_parameters=model_parameters, optimizer=optimizer, dist_init_required=False)
    return (deepspeed_engine, deepspeed_optimizer)