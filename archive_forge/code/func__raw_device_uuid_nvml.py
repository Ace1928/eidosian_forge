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
def _raw_device_uuid_nvml() -> Optional[List[str]]:
    """Return list of device UUID as reported by NVML or None if NVM discovery/initialization failed."""
    from ctypes import byref, c_int, c_void_p, CDLL, create_string_buffer
    nvml_h = CDLL('libnvidia-ml.so.1')
    rc = nvml_h.nvmlInit()
    if rc != 0:
        warnings.warn("Can't initialize NVML")
        return None
    dev_count = c_int(-1)
    rc = nvml_h.nvmlDeviceGetCount_v2(byref(dev_count))
    if rc != 0:
        warnings.warn("Can't get nvml device count")
        return None
    uuids: List[str] = []
    for idx in range(dev_count.value):
        dev_id = c_void_p()
        rc = nvml_h.nvmlDeviceGetHandleByIndex_v2(idx, byref(dev_id))
        if rc != 0:
            warnings.warn("Can't get device handle")
            return None
        buf_len = 96
        buf = create_string_buffer(buf_len)
        rc = nvml_h.nvmlDeviceGetUUID(dev_id, buf, buf_len)
        if rc != 0:
            warnings.warn("Can't get device UUID")
            return None
        uuids.append(buf.raw.decode('ascii').strip('\x00'))
    del nvml_h
    return uuids