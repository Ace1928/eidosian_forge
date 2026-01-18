import collections
import contextlib
import ctypes
import pickle
import sys
import warnings
from inspect import signature
from typing import Any, Dict, Optional, Tuple, Union
import torch
from torch import _C
from torch.types import Device
from . import _get_device_index, _get_nvml_device_index, _lazy_init, is_initialized
from ._memory_viz import memory as _memory, segments as _segments
from ._utils import _dummy_type
def _recurse_add_to_result(prefix, obj):
    if isinstance(obj, dict):
        if len(prefix) > 0:
            prefix += '.'
        for k, v in obj.items():
            _recurse_add_to_result(prefix + k, v)
    else:
        result.append((prefix, obj))