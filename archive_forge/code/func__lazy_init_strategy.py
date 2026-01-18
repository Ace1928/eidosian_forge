import logging
import os
from collections import Counter
from typing import Dict, List, Literal, Optional, Union
import torch
from lightning_fabric.connector import _PRECISION_INPUT, _PRECISION_INPUT_STR, _convert_precision_to_unified_args
from lightning_fabric.plugins.environments import (
from lightning_fabric.utilities.device_parser import _determine_root_gpu_device
from lightning_fabric.utilities.imports import _IS_INTERACTIVE
from pytorch_lightning.accelerators import AcceleratorRegistry
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.accelerators.cuda import CUDAAccelerator
from pytorch_lightning.accelerators.mps import MPSAccelerator
from pytorch_lightning.accelerators.xla import XLAAccelerator
from pytorch_lightning.plugins import (
from pytorch_lightning.plugins.layer_sync import LayerSync, TorchSyncBatchNorm
from pytorch_lightning.strategies import (
from pytorch_lightning.strategies.ddp import _DDP_FORK_ALIASES
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import (
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
def _lazy_init_strategy(self) -> None:
    """Lazily set missing attributes on the previously instantiated strategy."""
    self.strategy.accelerator = self.accelerator
    if self.precision_plugin:
        self.strategy.precision_plugin = self.precision_plugin
    if self.checkpoint_io:
        self.strategy.checkpoint_io = self.checkpoint_io
    if hasattr(self.strategy, 'cluster_environment'):
        if self.strategy.cluster_environment is None:
            self.strategy.cluster_environment = self.cluster_environment
        self.cluster_environment = self.strategy.cluster_environment
    if hasattr(self.strategy, 'parallel_devices'):
        if self.strategy.parallel_devices:
            self._parallel_devices = self.strategy.parallel_devices
        else:
            self.strategy.parallel_devices = self._parallel_devices
    if hasattr(self.strategy, 'num_nodes'):
        self.strategy.num_nodes = self._num_nodes_flag
    if hasattr(self.strategy, '_layer_sync'):
        self.strategy._layer_sync = self._layer_sync
    if hasattr(self.strategy, 'set_world_ranks'):
        self.strategy.set_world_ranks()
    self.strategy._configure_launcher()
    if _IS_INTERACTIVE and self.strategy.launcher and (not self.strategy.launcher.is_interactive_compatible):
        raise MisconfigurationException(f"`Trainer(strategy={self._strategy_flag!r})` is not compatible with an interactive environment. Run your code as a script, or choose a notebook-compatible strategy: `Trainer(strategy='ddp_notebook')`. In case you are spawning processes yourself, make sure to include the Trainer creation inside the worker function.")
    if isinstance(self.accelerator, XLAAccelerator) and (not isinstance(self.strategy, (SingleDeviceXLAStrategy, XLAStrategy))):
        raise ValueError(f'The `XLAAccelerator` can only be used with a `SingleDeviceXLAStrategy` or `XLAStrategy`, found {self.strategy.__class__.__name__}.')
    if _habana_available_and_importable():
        from lightning_habana import HPUAccelerator, HPUParallelStrategy, SingleHPUStrategy
        if isinstance(self.accelerator, HPUAccelerator) and (not isinstance(self.strategy, (SingleHPUStrategy, HPUParallelStrategy))):
            raise ValueError(f'The `HPUAccelerator` can only be used with a `SingleHPUStrategy` or `HPUParallelStrategy`, found {self.strategy.__class__.__name__}.')