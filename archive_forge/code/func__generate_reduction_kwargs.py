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
def _generate_reduction_kwargs(ndim, supports_multiple_dims=True):
    """Generates a subset of all valid dim and keepdim kwargs given ndim that
    is appropriate for testing reduction operators.
    """
    yield {}
    yield {'dim': 0, 'keepdim': True}
    yield {'dim': -1, 'keepdim': False}
    if ndim > 2:
        yield {'dim': ndim // 2, 'keepdim': True}
    if supports_multiple_dims:
        yield {'dim': tuple(range(ndim)), 'keepdim': False}
        if ndim > 1:
            yield {'dim': (0, -1), 'keepdim': True}
        if ndim > 3:
            yield {'dim': tuple(range(1, ndim, 2)), 'keepdim': False}