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
def _validate_precision_choice(self) -> None:
    """Validate the combination of choices for precision, AMP type, and accelerator."""
    if isinstance(self._precision_plugin_flag, BitsandbytesPrecision) and (not isinstance(self.accelerator, CUDAAccelerator)):
        raise RuntimeError('Bitsandbytes is only supported on CUDA GPUs.')
    if _habana_available_and_importable():
        from lightning_habana import HPUAccelerator
        if isinstance(self.accelerator, HPUAccelerator) and self._precision_flag not in ('16-mixed', 'bf16-mixed', '32-true'):
            raise MisconfigurationException(f"`Trainer(accelerator='hpu', precision={self._precision_flag!r})` is not supported.")