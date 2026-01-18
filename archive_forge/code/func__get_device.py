import contextlib
import importlib
import os
import sys
import threading
import traceback
import warnings
from functools import lru_cache
from typing import Any, cast, List, Optional, Tuple, Union
import torch
import torch._C
from torch.types import Device
from .. import device as _device
from .._utils import classproperty
from ._utils import _dummy_type, _get_device_index
from .graphs import (
from .streams import Event, ExternalStream, Stream
from .memory import *  # noqa: F403
from .random import *  # noqa: F403
from torch.storage import _LegacyStorage, _warn_typed_storage_removal
from . import amp, jiterator, nvtx, profiler, sparse
def _get_device(device: Union[int, str, torch.device]) -> torch.device:
    """Return the torch.device type object from the passed in device.

    Args:
        device (torch.device or int): selected device.
    """
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device('cuda', device)
    return device