import contextlib
import platform
import uuid
import warnings
import weakref
from collections import defaultdict
from itertools import count
from typing import (
from weakref import ReferenceType
import torch
import torch.fx.traceback as fx_traceback
from torch.utils._pytree import tree_map
from torch.testing._internal.logging_tensor import capture_logs, LoggingTensorMode
from torch.utils._python_dispatch import TorchDispatchMode
class DefaultDeviceType:
    """
    A class that manages the default device type for checkpointing.

    If no non-CPU tensors are present, the default device type will
    be used. The default value is 'cuda'. The device type is used in
    the checkpointing process when determining which device states
    to save and restore for recomputation.
    """
    _default_device_type = 'cuda'

    @staticmethod
    def set_device_type(device: str='cuda'):
        """
        Set the default device type for checkpointing.

        Args:
            device (str): The device type to be set as default. Default is 'cuda'.
        """
        DefaultDeviceType._default_device_type = device

    @staticmethod
    def get_device_type() -> str:
        """
        Get the current default device type for checkpointing.

        Returns:
            str: The current default device type.
        """
        return DefaultDeviceType._default_device_type