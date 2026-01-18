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
def _compute_sizes(seq, scalar_type):
    MAX_DIMS = 128
    is_storage = _isStorage(seq)
    sizes = []
    while isinstance(seq, (list, tuple)):
        length = len(seq)
        if is_storage:
            length //= scalar_type.itemsize
        sizes.append(length)
        if len(sizes) > MAX_DIMS:
            raise ValueError(f"too many dimensions '{type(seq).__name__}'")
        if length == 0:
            break
        try:
            handle = seq[0]
        except Exception:
            raise ValueError(f"could not determine the shape of object type '{type(seq).__name__}'")
        seq = handle
    return sizes