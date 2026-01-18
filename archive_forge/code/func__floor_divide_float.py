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
def _floor_divide_float(a: Tensor, b: Tensor) -> Tensor:
    mod = fmod(a, b)
    div = true_divide(sub(a, mod), b)
    different_signed_inputs = bitwise_xor(lt(a, 0), lt(b, 0))
    non_zero_remainder = ne(mod, 0)
    mask = bitwise_and(non_zero_remainder, different_signed_inputs)
    div = where(mask, sub(div, 1), div)
    floor_div = floor(div)
    mask = gt(sub(div, floor_div), 0.5)
    floor_div = where(mask, add(floor_div, 1), floor_div)
    basic_div = true_divide(a, b)
    zero_tensor = scalar_tensor(0, dtype=basic_div.dtype, device=basic_div.device)
    floor_div = where(ne(div, 0), floor_div, copysign(zero_tensor, basic_div))
    return where(ne(b, 0), floor_div, basic_div)