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
def _clear_cuda_memory() -> None:
    if _TORCH_GREATER_EQUAL_2_0 and hasattr(torch._C, '_cuda_clearCublasWorkspaces'):
        torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()