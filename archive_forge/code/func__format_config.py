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
def _format_config(self) -> None:
    if self.config is None:
        raise ValueError('To use DeepSpeed you must pass in a DeepSpeed config dict, or a path to a JSON config. See: https://lightning.ai/docs/pytorch/stable/advanced/model_parallel.html#deepspeed')
    self.config.setdefault('train_micro_batch_size_per_gpu', 1)
    _format_precision_config(config=self.config, precision=self.precision.precision, loss_scale=self.loss_scale, loss_scale_window=self.loss_scale_window, min_loss_scale=self.min_loss_scale, initial_scale_power=self.initial_scale_power, hysteresis=self.hysteresis)