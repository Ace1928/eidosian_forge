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
def _uniform_helper(shape: ShapeType, low: Union[bool, int, float]=0.0, high: Union[bool, int, float]=1.0, *, dtype: torch.dtype, device: DeviceLikeType) -> TensorLikeType:
    utils.validate_shape(shape)
    assert isinstance(low, Number)
    assert isinstance(high, Number)
    low = sym_float(low)
    high = sym_float(high)
    assert isinstance(dtype, torch.dtype)
    device = utils.canonicalize_device(device)
    return prims._uniform_helper(shape, low=low, high=high, dtype=dtype, device=device)