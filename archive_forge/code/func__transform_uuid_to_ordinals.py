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
def _transform_uuid_to_ordinals(candidates: List[str], uuids: List[str]) -> List[int]:
    """Given the set of partial uuids and list of known uuids builds a set of ordinals excluding ambiguous partials IDs."""

    def uuid_to_orinal(candidate: str, uuids: List[str]) -> int:
        best_match = -1
        for idx, uuid in enumerate(uuids):
            if not uuid.startswith(candidate):
                continue
            if best_match != -1:
                return -1
            best_match = idx
        return best_match
    rc: List[int] = []
    for candidate in candidates:
        idx = uuid_to_orinal(candidate, uuids)
        if idx < 0:
            break
        if idx in rc:
            return cast(List[int], [])
        rc.append(idx)
    return rc