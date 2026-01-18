import platform
from functools import lru_cache
from typing import List, Optional, Union
import torch
from typing_extensions import override
from lightning_fabric.accelerators.accelerator import Accelerator
from lightning_fabric.accelerators.registry import _AcceleratorRegistry
def _get_all_available_mps_gpus() -> List[int]:
    """
    Returns:
        A list of all available MPS GPUs
    """
    return [0] if MPSAccelerator.is_available() else []