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
class ErrorInput:
    """
    A SampleInput that will cause the operation to throw an error plus information
    about the resulting error.
    """
    __slots__ = ['sample_input', 'error_type', 'error_regex']

    def __init__(self, sample_input, *, error_type=RuntimeError, error_regex):
        self.sample_input = sample_input
        self.error_type = error_type
        self.error_regex = error_regex