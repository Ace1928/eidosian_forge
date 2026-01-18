import os
import warnings
from contextlib import contextmanager
from functools import lru_cache
from typing import Generator, List, Optional, Union, cast
import torch
from typing_extensions import override
from lightning_fabric.accelerators.accelerator import Accelerator
from lightning_fabric.accelerators.registry import _AcceleratorRegistry
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from lightning_fabric.utilities.rank_zero import rank_zero_info
def _get_all_visible_cuda_devices() -> List[int]:
    """Returns a list of all visible CUDA GPU devices.

    Devices masked by the environment variabale ``CUDA_VISIBLE_DEVICES`` won't be returned here. For example, assume you
    have 8 physical GPUs. If ``CUDA_VISIBLE_DEVICES="1,3,6"``, then this function will return the list ``[0, 1, 2]``
    because these are the three visible GPUs after applying the mask ``CUDA_VISIBLE_DEVICES``.

    """
    return list(range(num_cuda_devices()))