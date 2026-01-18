import logging
import traceback
import warnings
import weakref
from enum import auto, Enum
from functools import partial
from typing import (
import torch
import torch.distributed as dist
import torch.distributed.fsdp._flat_param as flat_param_file
import torch.nn as nn
from torch.distributed._composable_state import _get_module_state, _State
from torch.distributed._tensor.device_mesh import DeviceMesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
from torch.distributed.fsdp._fsdp_extensions import FSDPExtensions
from torch.distributed.utils import _apply_to_tensors
from torch.utils._mode_utils import no_dispatch
from .api import (
class _FSDPDeviceHandle:
    """
    This is a simple abstraction for FSDP computing devices,
    which enables custom backends that implement CUDA-like
    semantics to be integrated with FSDP.
    """

    def __init__(self, device: torch.device, backend: Any=None):
        if backend is None:
            try:
                self.__backend = getattr(torch, device.type)
                self.__device = device
            except AttributeError as exc:
                raise AttributeError(f"Device '{device}' does not have a corresponding backend registered as 'torch.{device.type}'.") from exc
        else:
            self.__backend = backend

    @classmethod
    def from_device(cls, device: torch.device) -> '_FSDPDeviceHandle':
        """
        Return an device handle corresponding to the device, and through this handle,
        operations with the same semantics as CUDA can be performed on the device.
        Just return torch.cuda if the device is cuda to make attribute-access faster.
        Custom backend must first register a module with the same name with {device.type} on torch.
        """
        if device.type == 'cuda':
            return cast(_FSDPDeviceHandle, torch.cuda)
        return cls(device)

    def __getattr__(self, __name: str) -> Any:
        try:
            return getattr(self.__backend, __name)
        except AttributeError as exc:
            raise AttributeError(f"Custom backend '{self.__device.type}' not implement 'torch.{self.__device.type}.{__name}'") from exc