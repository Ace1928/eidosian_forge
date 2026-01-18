import logging
import os
import re
import subprocess
import sys
from argparse import Namespace
from typing import Any, List, Optional
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import get_args
from lightning_fabric.accelerators import CPUAccelerator, CUDAAccelerator, MPSAccelerator
from lightning_fabric.plugins.precision.precision import _PRECISION_INPUT_STR, _PRECISION_INPUT_STR_ALIAS
from lightning_fabric.strategies import STRATEGY_REGISTRY
from lightning_fabric.utilities.device_parser import _parse_gpu_ids
from lightning_fabric.utilities.distributed import _suggested_max_num_threads
def _get_supported_strategies() -> List[str]:
    """Returns strategy choices from the registry, with the ones removed that are incompatible to be launched from the
    CLI or ones that require further configuration by the user."""
    available_strategies = STRATEGY_REGISTRY.available_strategies()
    excluded = '.*(spawn|fork|notebook|xla|tpu|offload).*'
    return [strategy for strategy in available_strategies if not re.match(excluded, strategy)]