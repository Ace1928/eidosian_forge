import dataclasses
import warnings
from contextlib import nullcontext
from functools import wraps
from typing import Any, Callable, Optional, Tuple
import torch
import torch.utils._pytree as pytree
from torch.fx.experimental.proxy_tensor import py_sym_types
def _get_autocast_states():
    return [torch.is_autocast_enabled(), torch.is_autocast_cpu_enabled(), torch.get_autocast_gpu_dtype(), torch.get_autocast_cpu_dtype(), torch.is_autocast_cache_enabled()]