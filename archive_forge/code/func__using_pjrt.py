import functools
from typing import Any, List, Union
import torch
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override
from lightning_fabric.accelerators.accelerator import Accelerator
from lightning_fabric.accelerators.registry import _AcceleratorRegistry
from lightning_fabric.utilities.device_parser import _check_data_type
def _using_pjrt() -> bool:
    if _XLA_GREATER_EQUAL_2_1:
        from torch_xla import runtime as xr
        return xr.using_pjrt()
    from torch_xla.experimental import pjrt
    return pjrt.using_pjrt()