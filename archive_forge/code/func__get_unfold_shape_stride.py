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
def _get_unfold_shape_stride(a_shape: ShapeType, a_stride: StrideType, dimension: int, size: int, step: int):
    a_ndim = len(a_shape)
    dim = utils.canonicalize_dim(a_ndim, dimension, wrap_scalar=True)
    max_size = 1 if a_ndim == 0 else a_shape[dim]
    last_stride = 1 if a_ndim == 0 else a_stride[dim]
    torch._check(size <= max_size, lambda: f'Maximum size for tensor at dimension {dim} is {max_size} but size is {size}')
    torch._check(step > 0, lambda: f'Step is {step} but must be > 0')
    shape = list(a_shape)
    strides = list(a_stride)
    shape.append(size)
    strides.append(last_stride)
    if dim < a_ndim:
        shape[dim] = (shape[dim] - size) // step + 1
        strides[dim] *= step
    return (shape, strides)