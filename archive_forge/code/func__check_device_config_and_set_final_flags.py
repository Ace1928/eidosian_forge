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
def _check_device_config_and_set_final_flags(self, devices: Union[List[int], str, int], num_nodes: int) -> None:
    if not isinstance(num_nodes, int) or num_nodes < 1:
        raise ValueError(f'`num_nodes` must be a positive integer, but got {num_nodes}.')
    self._num_nodes_flag = num_nodes
    self._devices_flag = devices
    if self._devices_flag in ([], 0, '0'):
        accelerator_name = self._accelerator_flag.__class__.__qualname__ if isinstance(self._accelerator_flag, Accelerator) else self._accelerator_flag
        raise MisconfigurationException(f'`Trainer(devices={self._devices_flag!r})` value is not a valid input using {accelerator_name} accelerator.')