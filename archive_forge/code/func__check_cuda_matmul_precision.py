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
@lru_cache(1)
def _check_cuda_matmul_precision(device: torch.device) -> None:
    if not torch.cuda.is_available() or not _is_ampere_or_later(device):
        return
    if torch.get_float32_matmul_precision() == 'highest':
        rank_zero_info(f"You are using a CUDA device ({torch.cuda.get_device_name(device)!r}) that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision")