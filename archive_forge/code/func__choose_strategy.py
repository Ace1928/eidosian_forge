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
def _choose_strategy(self) -> Union[Strategy, str]:
    if self._accelerator_flag == 'ipu':
        if not _graphcore_available_and_importable():
            raise ImportError("You have passed `accelerator='ipu'` but the IPU integration  is not installed. Please run `pip install lightning-graphcore` or check out https://github.com/Lightning-AI/lightning-Graphcore for instructions")
        from lightning_graphcore import IPUStrategy
        return IPUStrategy.strategy_name
    if _habana_available_and_importable():
        from lightning_habana import HPUAccelerator
        if self._accelerator_flag == 'hpu' or isinstance(self._accelerator_flag, HPUAccelerator):
            if self._parallel_devices and len(self._parallel_devices) > 1:
                from lightning_habana import HPUParallelStrategy
                return HPUParallelStrategy.strategy_name
            from lightning_habana import SingleHPUStrategy
            return SingleHPUStrategy(device=torch.device('hpu'))
    if self._accelerator_flag == 'hpu' and (not _habana_available_and_importable()):
        raise ImportError('You asked to run with HPU but you are missing a required dependency. Please run `pip install lightning-habana` or seek further instructions in https://github.com/Lightning-AI/lightning-Habana/.')
    if self._accelerator_flag == 'tpu' or isinstance(self._accelerator_flag, XLAAccelerator):
        if self._parallel_devices and len(self._parallel_devices) > 1:
            return XLAStrategy.strategy_name
        return SingleDeviceXLAStrategy(device=self._parallel_devices[0])
    if self._num_nodes_flag > 1:
        return 'ddp'
    if len(self._parallel_devices) <= 1:
        if isinstance(self._accelerator_flag, (CUDAAccelerator, MPSAccelerator)) or (isinstance(self._accelerator_flag, str) and self._accelerator_flag in ('cuda', 'gpu', 'mps')):
            device = _determine_root_gpu_device(self._parallel_devices)
        else:
            device = 'cpu'
        return SingleDeviceStrategy(device=device)
    if len(self._parallel_devices) > 1 and _IS_INTERACTIVE:
        return 'ddp_fork'
    return 'ddp'