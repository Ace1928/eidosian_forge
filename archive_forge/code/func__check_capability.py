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
def _check_capability():
    incorrect_binary_warn = '\n    Found GPU%d %s which requires CUDA_VERSION >= %d to\n     work properly, but your PyTorch was compiled\n     with CUDA_VERSION %d. Please install the correct PyTorch binary\n     using instructions from https://pytorch.org\n    '
    old_gpu_warn = '\n    Found GPU%d %s which is of cuda capability %d.%d.\n    PyTorch no longer supports this GPU because it is too old.\n    The minimum cuda capability supported by this library is %d.%d.\n    '
    if torch.version.cuda is not None:
        CUDA_VERSION = torch._C._cuda_getCompiledVersion()
        for d in range(device_count()):
            capability = get_device_capability(d)
            major = capability[0]
            minor = capability[1]
            name = get_device_name(d)
            current_arch = major * 10 + minor
            min_arch = min((int(arch.split('_')[1]) for arch in torch.cuda.get_arch_list()), default=35)
            if current_arch < min_arch:
                warnings.warn(old_gpu_warn % (d, name, major, minor, min_arch // 10, min_arch % 10))