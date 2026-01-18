import math
import functools
import warnings
from collections import OrderedDict, defaultdict
from copy import deepcopy
from itertools import chain
from typing import (
from typing_extensions import ParamSpec, Self, TypeAlias
import torch
import torch.utils.hooks as hooks
from torch.utils.hooks import RemovableHandle
from torch.utils._foreach_utils import (
from torch._utils import is_compiling
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype
def _default_to_fused_or_foreach(params: List[torch.Tensor], differentiable: bool, use_fused: bool=False) -> Tuple[bool, bool]:
    if torch.jit.is_scripting() or differentiable:
        return (False, False)
    fused_supported_devices = _get_fused_kernels_supported_devices()
    foreach_supported_devices = _get_foreach_kernels_supported_devices()
    fused = use_fused and all((p is None or (type(p) in _foreach_supported_types and p.device.type in fused_supported_devices and torch.is_floating_point(p)) for p in params))
    foreach = not fused and all((p is None or (type(p) in _foreach_supported_types and p.device.type in foreach_supported_devices) for p in params))
    return (fused, foreach)