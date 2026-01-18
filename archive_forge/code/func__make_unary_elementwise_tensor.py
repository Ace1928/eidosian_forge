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
def _make_unary_elementwise_tensor(shape, *, op, dtype, **kwargs):
    low, high = op.domain
    is_floating = dtype.is_floating_point or dtype.is_complex
    low = low if low is None or not is_floating else low + op._domain_eps
    high = high if high is None or not is_floating else high - op._domain_eps
    a = make_tensor(shape, low=low, high=high, dtype=dtype, **kwargs)
    if op.reference_numerics_filter is not None and dtype is not torch.bool:
        condition, safe_value = op.reference_numerics_filter
        _replace_values_in_tensor(a, condition, safe_value)
    return a