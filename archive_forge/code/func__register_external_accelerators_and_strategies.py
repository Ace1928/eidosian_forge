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
def _register_external_accelerators_and_strategies() -> None:
    """Registers all known strategies in other packages."""
    if _LIGHTNING_COLOSSALAI_AVAILABLE:
        from lightning_colossalai import ColossalAIStrategy
        if 'colossalai' not in StrategyRegistry:
            ColossalAIStrategy.register_strategies(StrategyRegistry)
    if _LIGHTNING_BAGUA_AVAILABLE:
        from lightning_bagua import BaguaStrategy
        if 'bagua' not in StrategyRegistry:
            BaguaStrategy.register_strategies(StrategyRegistry)
    if _habana_available_and_importable():
        from lightning_habana import HPUAccelerator, HPUParallelStrategy, SingleHPUStrategy
        if 'hpu' not in AcceleratorRegistry:
            HPUAccelerator.register_accelerators(AcceleratorRegistry)
        if 'hpu_parallel' not in StrategyRegistry:
            HPUParallelStrategy.register_strategies(StrategyRegistry)
        if 'hpu_single' not in StrategyRegistry:
            SingleHPUStrategy.register_strategies(StrategyRegistry)
    if _graphcore_available_and_importable():
        from lightning_graphcore import IPUAccelerator, IPUStrategy
        if 'ipu' not in AcceleratorRegistry:
            IPUAccelerator.register_accelerators(AcceleratorRegistry)
        if 'ipu_strategy' not in StrategyRegistry:
            IPUStrategy.register_strategies(StrategyRegistry)