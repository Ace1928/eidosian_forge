import functools
from typing import Any, List, Union
import torch
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override
from lightning_fabric.accelerators.accelerator import Accelerator
from lightning_fabric.accelerators.registry import _AcceleratorRegistry
from lightning_fabric.utilities.device_parser import _check_data_type
def _check_tpu_devices_valid(devices: object) -> None:
    device_count = XLAAccelerator.auto_device_count()
    if isinstance(devices, int) and devices in {1, device_count} or (isinstance(devices, (list, tuple)) and len(devices) == 1 and (0 <= devices[0] <= device_count - 1)):
        return
    raise ValueError(f"`devices` can only be 'auto', 1, {device_count} or [<0-{device_count - 1}>] for TPUs. Got {devices!r}")