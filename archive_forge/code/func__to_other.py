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
@_to_dispatch.register
def _to_other(other: Tensor, non_blocking: bool=False, copy: bool=False, memory_format: Optional[torch.memory_format]=None) -> Dict[str, Any]:
    device = other.device
    dtype = other.dtype
    layout = other.layout
    kwargs = {'device': device, 'dtype': dtype, 'layout': layout, 'non_blocking': non_blocking, 'copy': copy, 'memory_format': memory_format}
    return kwargs