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
def _strtoul(s: str) -> int:
    """Return -1 or positive integer sequence string starts with."""
    if not s:
        return -1
    for idx, c in enumerate(s):
        if not (c.isdigit() or (idx == 0 and c in '+-')):
            break
        if idx + 1 == len(s):
            idx += 1
    return int(s[:idx]) if idx > 0 else -1