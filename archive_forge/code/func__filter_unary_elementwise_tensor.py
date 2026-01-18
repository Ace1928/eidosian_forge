import collections
import collections.abc
import math
import operator
import unittest
from dataclasses import asdict, dataclass
from enum import Enum
from functools import partial
from itertools import product
from typing import Any, Callable, Iterable, List, Optional, Tuple
from torchgen.utils import dataclass_repr
import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_utils import (
from torch.testing._internal.opinfo import utils
def _filter_unary_elementwise_tensor(a, *, op):
    if a.dtype is torch.bool:
        return a
    low, high = op.domain
    is_floating = a.dtype.is_floating_point or a.dtype.is_complex
    low = low if low is None or not is_floating else low + op._domain_eps
    high = high if high is None or not is_floating else high - op._domain_eps
    if a.dtype is torch.uint8 and low is not None:
        low = max(low, 0)
    if not a.dtype.is_floating_point and (not a.dtype.is_complex):
        low = math.ceil(low) if low is not None else None
        high = math.floor(high) if high is not None else None
    if op.reference_numerics_filter is not None:
        condition, safe_value = op.reference_numerics_filter
        _replace_values_in_tensor(a, condition, safe_value)
    if low is not None or high is not None:
        if a.dtype.is_complex:
            a.real.clamp_(low, high)
            a.imag.clamp_(low, high)
        else:
            a.clamp_(min=low, max=high)
    return a