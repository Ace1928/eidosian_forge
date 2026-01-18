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
def _canonicalize_to_arguments(a: Tensor, to_kwargs: dict):
    options_to_check = ['dtype', 'device', 'layout', 'memory_format']
    if 'device' in to_kwargs and isinstance(to_kwargs['device'], str):
        to_kwargs['device'] = torch.device(to_kwargs['device'])
    for kw in options_to_check:
        if kw in to_kwargs:
            if kw == 'memory_format' and to_kwargs[kw] is torch.preserve_format or (kw == 'device' and to_kwargs[kw].type == a.device.type and (not to_kwargs[kw].index or to_kwargs[kw].index == a.device.index)) or getattr(a, kw, None) == to_kwargs[kw]:
                to_kwargs.pop(kw)