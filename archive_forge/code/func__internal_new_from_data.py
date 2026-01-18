import builtins
import collections
import inspect
import itertools
import math
import operator
import warnings
from collections.abc import Iterable
from enum import Enum
from functools import partial, reduce, singledispatch, wraps
from typing import Any, Callable, Dict, List, Optional, overload, Sequence, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
from torch import sym_float, sym_int
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._decomp import register_decomposition
import torch._refs._conversions
import torch._refs.fft
import torch._refs.linalg
import torch._refs.nn.functional
import torch._refs.special
def _internal_new_from_data(options, scalar_type, device_opt, data, copy_variables, copy_numpy, type_inference, pin_memory=False):
    if isinstance(data, torch.Tensor):
        torch._check(not pin_memory, lambda: "Can't pin tensor constructed from a variable")
        var = data
        if copy_variables:
            var = var.detach()
        inferred_scalar_type = var.dtype if type_inference else scalar_type
        device = device_opt if device_opt is not None else var.device
        return var.to(device=device, dtype=inferred_scalar_type, non_blocking=False, copy=copy_variables)
    if hasattr(data, '__cuda_array_interface__'):
        return NotImplemented
    device = device_opt if device_opt is not None else options['device']
    sizes = _compute_sizes(data, scalar_type)
    inferred_scalar_type = _infer_scalar_type(data) if type_inference else scalar_type
    if _isStorage(data):
        return NotImplemented
    else:
        if torch.device(device).type == 'meta':
            return NotImplemented
        tensor = _recursive_build(sizes, 0, inferred_scalar_type, data)
        tensor = tensor.to(device, inferred_scalar_type, non_blocking=False, copy=False)
    return tensor