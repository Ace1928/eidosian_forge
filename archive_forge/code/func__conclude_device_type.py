from __future__ import annotations, division
import ast
import functools
import hashlib
import inspect
import os
import textwrap
from collections import defaultdict, namedtuple
from functools import cached_property
from typing import Callable, Generic, Iterable, List, Optional, TypeVar, Union, cast, overload
from .._C.libtriton.triton import TMAInfos
from ..common.backend import get_backend, get_cuda_version_key
from .interpreter import InterpretedFunction
def _conclude_device_type(self, device_types: List[str], pinned_memory_flags: List[bool]) -> str:
    device_types = [device_type for device_type in device_types if device_type != '']
    if 'cuda' in device_types:
        import torch
        return 'hip' if torch.version.hip else 'cuda'
    is_cpu = all((device_type == 'cpu' for device_type in device_types))
    is_pinned_memory = any((pinned_memory_flag for pinned_memory_flag in pinned_memory_flags))
    if is_cpu and is_pinned_memory:
        return 'cuda'
    return device_types[0] if len(device_types) > 0 else 'cuda'