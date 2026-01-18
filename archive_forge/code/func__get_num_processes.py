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
def _get_num_processes(accelerator: str, devices: str) -> int:
    """Parse the `devices` argument to determine how many processes need to be launched on the current machine."""
    if accelerator == 'gpu':
        parsed_devices = _parse_gpu_ids(devices, include_cuda=True, include_mps=True)
    elif accelerator == 'cuda':
        parsed_devices = CUDAAccelerator.parse_devices(devices)
    elif accelerator == 'mps':
        parsed_devices = MPSAccelerator.parse_devices(devices)
    elif accelerator == 'tpu':
        raise ValueError('Launching processes for TPU through the CLI is not supported.')
    else:
        return CPUAccelerator.parse_devices(devices)
    return len(parsed_devices) if parsed_devices is not None else 0