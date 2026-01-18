from __future__ import annotations
import inspect
import warnings
from collections import abc, defaultdict
from enum import Enum
from typing import Any, cast, Dict, Iterable, List, Optional, overload, Tuple, Union
import torch
from .common import amp_definitely_not_available
class _MultiDeviceReplicator:
    """Lazily serves copies of a tensor to requested devices.

    Copies are cached per-device.
    """

    def __init__(self, master_tensor: torch.Tensor) -> None:
        assert master_tensor.is_cuda or master_tensor.device.type == 'xla'
        self.master = master_tensor
        self._per_device_tensors: Dict[torch.device, torch.Tensor] = {}

    def get(self, device: torch.device) -> torch.Tensor:
        retval = self._per_device_tensors.get(device, None)
        if retval is None:
            retval = self.master.to(device=device, non_blocking=True, copy=True)
            self._per_device_tensors[device] = retval
        return retval