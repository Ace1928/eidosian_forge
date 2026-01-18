import os
from collections import Counter
from typing import Any, Dict, List, Optional, Union, cast
import torch
from typing_extensions import get_args
from lightning_fabric.accelerators import ACCELERATOR_REGISTRY
from lightning_fabric.accelerators.accelerator import Accelerator
from lightning_fabric.accelerators.cuda import CUDAAccelerator
from lightning_fabric.accelerators.mps import MPSAccelerator
from lightning_fabric.accelerators.xla import XLAAccelerator
from lightning_fabric.plugins import (
from lightning_fabric.plugins.environments import (
from lightning_fabric.plugins.precision.double import DoublePrecision
from lightning_fabric.plugins.precision.fsdp import FSDPPrecision
from lightning_fabric.plugins.precision.precision import (
from lightning_fabric.strategies import (
from lightning_fabric.strategies.ddp import _DDP_FORK_ALIASES
from lightning_fabric.strategies.fsdp import _FSDP_ALIASES, FSDPStrategy
from lightning_fabric.utilities import rank_zero_info, rank_zero_warn
from lightning_fabric.utilities.device_parser import _determine_root_gpu_device
from lightning_fabric.utilities.imports import _IS_INTERACTIVE
def _convert_precision_to_unified_args(precision: Optional[_PRECISION_INPUT]) -> Optional[_PRECISION_INPUT_STR]:
    if precision is None:
        return None
    supported_precision = get_args(_PRECISION_INPUT_STR) + get_args(_PRECISION_INPUT_INT) + get_args(_PRECISION_INPUT_STR_ALIAS)
    if precision not in supported_precision:
        raise ValueError(f'Precision {repr(precision)} is invalid. Allowed precision values: {supported_precision}')
    precision = str(precision)
    if precision in get_args(_PRECISION_INPUT_STR_ALIAS):
        if str(precision)[:2] not in ('32', '64'):
            rank_zero_warn(f'`precision={precision}` is supported for historical reasons but its usage is discouraged. Please set your precision to {_PRECISION_INPUT_STR_ALIAS_CONVERSION[precision]} instead!')
        precision = _PRECISION_INPUT_STR_ALIAS_CONVERSION[precision]
    return cast(_PRECISION_INPUT_STR, precision)